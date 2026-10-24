#include <ATen/Context.h>
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_fused_adamw.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

struct Args {
  std::string model_path;
  std::string data_path;
  std::string output_path;
  int64_t epochs;
  int64_t batch_size;
  double learning_rate;
  double weight_decay;
  uint64_t seed;
  std::string device_text;
  torch::Device device;
};

static void usage(std::ostream& out) {
  out << "usage: native model.pt data.pt output.pt epochs batch lr wd seed device\n"
      << "device is cuda, cuda:<index>, or cpu (cpu is for verification only)\n";
}

static int64_t parse_i64(const std::string& text, const char* name) {
  size_t end = 0;
  try {
    const auto value = std::stoll(text, &end);
    if (end != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
  }
}

static uint64_t parse_u64(const std::string& text, const char* name) {
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
  }
  size_t end = 0;
  try {
    const auto value = std::stoull(text, &end);
    if (end != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
  }
}

static double parse_double(const std::string& text, const char* name) {
  size_t end = 0;
  try {
    const auto value = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(value)) {
      throw std::invalid_argument("not finite");
    }
    return value;
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
  }
}

static torch::Device parse_device(const std::string& text) {
  if (text == "cpu") {
    return torch::Device(torch::kCPU);
  }
  if (text == "cuda") {
    return torch::Device(torch::kCUDA, 0);
  }
  constexpr const char prefix[] = "cuda:";
  if (text.rfind(prefix, 0) == 0) {
    const auto index = parse_i64(text.substr(sizeof(prefix) - 1), "device index");
    if (index < 0 || index > std::numeric_limits<int16_t>::max()) {
      throw std::invalid_argument("invalid cuda device index: " + text);
    }
    return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(index));
  }
  throw std::invalid_argument("device must be cuda, cuda:<index>, or cpu");
}

static Args parse_args(int argc, char** argv) {
  if (argc != 10) {
    usage(std::cerr);
    throw std::invalid_argument("expected exactly nine positional arguments");
  }
  Args args{argv[1], argv[2], argv[3], parse_i64(argv[4], "epochs"),
            parse_i64(argv[5], "batch"), parse_double(argv[6], "lr"),
            parse_double(argv[7], "wd"), parse_u64(argv[8], "seed"), argv[9],
            parse_device(argv[9])};
  if (args.epochs <= 0 || args.batch_size <= 0) {
    throw std::invalid_argument("epochs and batch must be positive");
  }
  if (args.learning_rate < 0.0 || args.weight_decay < 0.0) {
    throw std::invalid_argument("lr and wd must be nonnegative");
  }
  return args;
}

static torch::Tensor required_buffer(
    const torch::jit::Module& data,
    const char* name) {
  TORCH_CHECK(data.hasattr(name), "data bundle is missing buffer '", name, "'");
  const auto value = data.attr(name);
  TORCH_CHECK(value.isTensor(), "data bundle buffer '", name, "' is not a Tensor");
  return value.toTensor();
}

static void validate_float_matrix(
    const torch::Tensor& value,
    const char* name,
    int64_t rows,
    int64_t columns) {
  TORCH_CHECK(value.device().is_cpu(), name, " must be CPU resident in the data bundle");
  TORCH_CHECK(value.scalar_type() == torch::kFloat32,
              name, " must have dtype float32; refusing an implicit cast");
  TORCH_CHECK(value.dim() == 2 && value.size(0) == rows && value.size(1) == columns,
              name, " must have shape [", rows, ", ", columns, "]");
}

static void configure_determinism(uint64_t seed) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
  torch::manual_seed(seed);
  if (torch::cuda::is_available()) {
    torch::cuda::manual_seed_all(seed);
  }
}

int main(int argc, char** argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" ||
                    std::string(argv[1]) == "-h")) {
    usage(std::cout);
    return 0;
  }

  try {
    const Args args = parse_args(argc, argv);
    TORCH_CHECK(!fs::exists(args.output_path), "refusing to overwrite existing output: ",
                args.output_path);
    TORCH_CHECK(fs::exists(args.model_path), "model does not exist: ", args.model_path);
    TORCH_CHECK(fs::exists(args.data_path), "data bundle does not exist: ", args.data_path);
    TORCH_CHECK(!args.device.is_cuda() || torch::cuda::is_available(),
                "CUDA device requested but CUDA is unavailable");
    if (args.device.is_cuda()) {
      TORCH_CHECK(args.device.index() < torch::cuda::device_count(),
                  "CUDA device index is unavailable: ", args.device_text);
    }
    configure_determinism(args.seed);

    auto data = torch::jit::load(args.data_path, torch::kCPU);
    auto features = required_buffer(data, "features").contiguous();
    auto targets = required_buffer(data, "targets").contiguous();
    auto weights = required_buffer(data, "weights").contiguous();
    TORCH_CHECK(features.dim() == 2 && features.size(0) > 0 && features.size(1) > 0,
                "features must have nonempty shape [N, D]");
    const int64_t rows = features.size(0);
    const int64_t columns = features.size(1);
    validate_float_matrix(features, "features", rows, columns);
    validate_float_matrix(targets, "targets", rows, 1);
    validate_float_matrix(weights, "weights", rows, 1);
    TORCH_CHECK(torch::isfinite(features).all().item<bool>(), "features contain non-finite values");

    const auto* target_ptr = targets.data_ptr<float>();
    const auto* weight_ptr = weights.data_ptr<float>();
    double total_weight = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
      TORCH_CHECK(std::isfinite(target_ptr[i]) &&
                      (target_ptr[i] == 0.0F || target_ptr[i] == 1.0F),
                  "targets must be finite binary values");
      TORCH_CHECK(std::isfinite(weight_ptr[i]) && weight_ptr[i] >= 0.0F,
                  "weights must be finite and nonnegative");
      total_weight += static_cast<double>(weight_ptr[i]);
    }
    TORCH_CHECK(std::isfinite(total_weight) && total_weight > 0.0,
                "weights must have a finite positive total");

    if (data.hasattr("order")) {
      auto order = required_buffer(data, "order").contiguous();
      TORCH_CHECK(order.device().is_cpu() && order.scalar_type() == torch::kInt64 &&
                      order.dim() == 1 && order.size(0) == rows,
                  "order must have dtype int64 and shape [N]");
      std::vector<uint8_t> seen(static_cast<size_t>(rows), 0);
      const auto* order_ptr = order.data_ptr<int64_t>();
      for (int64_t i = 0; i < rows; ++i) {
        TORCH_CHECK(order_ptr[i] >= 0 && order_ptr[i] < rows, "order is out of range");
        TORCH_CHECK(!seen[static_cast<size_t>(order_ptr[i])], "order must be a permutation");
        seen[static_cast<size_t>(order_ptr[i])] = 1;
      }
      features = features.index_select(0, order).contiguous();
      targets = targets.index_select(0, order).contiguous();
      weights = weights.index_select(0, order).contiguous();
    }
    if (args.device.is_cuda()) {
      features = features.pin_memory();
      targets = targets.pin_memory();
      weights = weights.pin_memory();
    }

    auto model = torch::jit::load(args.model_path, torch::kCPU);
    model.to(args.device);
    model.train();
    std::vector<torch::Tensor> trainable;
    for (const auto& parameter : model.parameters()) {
      if (parameter.requires_grad()) {
        TORCH_CHECK(parameter.scalar_type() == torch::kFloat32,
                    "model parameters must have dtype float32");
        trainable.push_back(parameter);
      }
    }
    TORCH_CHECK(!trainable.empty(), "model has no trainable parameters");
    torch::optim::AdamWOptions optim_options(args.learning_rate);
    optim_options.weight_decay(args.weight_decay);
    optim_options.betas(std::make_tuple(0.9, 0.999));
    optim_options.eps(1e-8);
    std::unique_ptr<torch::optim::AdamW> optimizer;
    std::vector<torch::Tensor> fused_exp_avg;
    std::vector<torch::Tensor> fused_exp_avg_sq;
    std::vector<torch::Tensor> fused_steps;
    std::vector<torch::Tensor> empty_max_exp_avg_sq;
    const std::string optimizer_backend =
        args.device.is_cuda() ? "fused_cpp" : "libtorch_adamw";
    if (args.device.is_cuda()) {
      for (const auto& parameter : trainable) {
        TORCH_CHECK(parameter.device().is_cuda() &&
                        parameter.scalar_type() == torch::kFloat32,
                    "fused CUDA optimizer requires float32 CUDA parameters");
        fused_exp_avg.push_back(torch::zeros_like(parameter).detach());
        fused_exp_avg_sq.push_back(torch::zeros_like(parameter).detach());
        fused_steps.push_back(torch::zeros(
            {}, parameter.options().dtype(torch::kFloat32).requires_grad(false)));
      }
    } else {
      optimizer = std::make_unique<torch::optim::AdamW>(trainable, optim_options);
    }

    if (args.device.is_cuda()) {
      torch::cuda::synchronize(args.device.index());
    }
    const auto started = std::chrono::steady_clock::now();
    torch::Tensor final_loss;
    for (int64_t epoch = 0; epoch < args.epochs; ++epoch) {
      auto epoch_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(args.device));
      for (int64_t start = 0; start < rows; start += args.batch_size) {
        const int64_t count = std::min(args.batch_size, rows - start);
        auto batch_features = features.narrow(0, start, count).to(args.device, true);
        auto batch_targets = targets.narrow(0, start, count).to(args.device, true);
        auto batch_weights = weights.narrow(0, start, count).to(args.device, true);
        if (optimizer) {
          optimizer->zero_grad();
        } else {
          for (auto& parameter : trainable) {
            if (parameter.grad().defined()) {
              parameter.mutable_grad() = torch::Tensor();
            }
          }
        }
        const auto output = model.forward({batch_features});
        TORCH_CHECK(output.isTensor(), "model forward must return a Tensor of logits");
        const auto logits = output.toTensor();
        TORCH_CHECK(logits.sizes() == batch_targets.sizes(),
                    "model logits must have shape [batch, 1]");
        auto loss_options = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
                                .weight(batch_weights)
                                .reduction(torch::kMean);
        const auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
            logits, batch_targets, loss_options);
        loss.backward();
        torch::nn::utils::clip_grad_norm_(trainable, 1.0, 2.0, true);
        if (optimizer) {
          optimizer->step();
        } else {
          std::vector<torch::Tensor> selected_params;
          std::vector<torch::Tensor> selected_grads;
          std::vector<torch::Tensor> selected_exp_avg;
          std::vector<torch::Tensor> selected_exp_avg_sq;
          std::vector<torch::Tensor> selected_steps;
          for (size_t i = 0; i < trainable.size(); ++i) {
            if (trainable[i].grad().defined()) {
              selected_params.push_back(trainable[i]);
              selected_grads.push_back(trainable[i].grad());
              selected_exp_avg.push_back(fused_exp_avg[i]);
              selected_exp_avg_sq.push_back(fused_exp_avg_sq[i]);
              selected_steps.push_back(fused_steps[i]);
            }
          }
          if (!selected_params.empty()) {
            torch::NoGradGuard no_grad;
            at::_foreach_add_(selected_steps, 1);
            at::_fused_adamw_(
                selected_params, selected_grads, selected_exp_avg,
                selected_exp_avg_sq, empty_max_exp_avg_sq, selected_steps,
                args.learning_rate, 0.9, 0.999, args.weight_decay, 1e-8,
                false, false);
          }
        }
        epoch_sum = epoch_sum + loss.detach() * static_cast<double>(count);
      }
      final_loss = epoch_sum / static_cast<double>(rows);
    }
    if (args.device.is_cuda()) {
      torch::cuda::synchronize(args.device.index());
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const double final_loss_value = final_loss.item<double>();
    TORCH_CHECK(std::isfinite(final_loss_value), "training produced a non-finite loss");
    for (const auto& parameter : trainable) {
      TORCH_CHECK(torch::isfinite(parameter).all().item<bool>(),
                  "training produced a non-finite parameter");
    }

    TORCH_CHECK(!fs::exists(args.output_path), "refusing to overwrite existing output: ",
                args.output_path);
    model.eval();
    model.to(torch::Device(torch::kCPU));
    model.save(args.output_path);
    std::cout << std::setprecision(17)
              << "{\"status\":\"ok\",\"rows\":" << rows
              << ",\"features\":" << columns
              << ",\"epochs\":" << args.epochs
              << ",\"batch\":" << args.batch_size
              << ",\"lr\":" << args.learning_rate
              << ",\"wd\":" << args.weight_decay
              << ",\"seed\":" << args.seed
              << ",\"device\":\"" << args.device_text
              << "\",\"optimizer_backend\":\"" << optimizer_backend
              << "\",\"loss\":\"mean(weight*bce_with_logits)\""
              << ",\"grad_clip_max_norm\":1"
              << ",\"final_loss\":" << final_loss_value
              << ",\"training_seconds\":" << elapsed << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native trainer error: " << error.what() << '\n';
    return 1;
  }
}
