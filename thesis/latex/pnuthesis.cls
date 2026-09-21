%% pnuthesis.cls — 부산대학교 대학원 학위논문 LaTeX 클래스 (비공식)
%%
%% 근거 규격:
%%   * 부산대학교 학생지원시스템 > 졸업 > 학위청구논문 > 논문 작성 지침
%%     https://onestop.pusan.ac.kr/page?menuCD=000000000000302
%%   * 첨부 「학위논문작성가이드(국문석사).hwp」 (thesis/forms/ 에 사본)
%%       - 용지 A4(210x297 mm), 여백: 위 35 / 아래 25 / 왼쪽 25(+제본 5) / 오른쪽 25 mm, 꼬리말 15 mm
%%       - 제본 순서: 표지 → 면지 → 속표지 및 인준지 → 목차 → 초록(본문 언어) → 본문 → 참고문헌 → 부록 → 초록(다른 언어)
%%       - 표지: "석 사 학 위 논 문"(16 pt) / 제목(22 pt) / 성명(16) / 부산대학교 대학원(16) / OOO과(16) / 학위수여년월(16)
%%       - 속표지·인준지: 제목(22) / "이 논문을 OO석사 학위논문으로 제출함"(16) / 성명·대학원·학과·지도교수(14)
%%                        / "OOO의 OO석사 학위논문을 인준함"(16) / 연월일(14) / 위원장·위원·위원(14)
%%       - 초록: 제목(14) / 성명(12) / 부산대학교 대학원 OOO과(11) / 요약(12) / 본문(11)
%%       - 본문·참고문헌·부록의 글꼴·크기·줄간격은 지도교수 지도에 따라 자유 (본 클래스: 11 pt, 줄간격 160%)
%%   * 규정은 바뀔 수 있으므로 최종 제출 전 반드시 100% 인쇄 후 자로 확인할 것.
%%
%% 사용: \documentclass[master,korean]{pnuthesis}
%% 엔진: XeLaTeX (kotex/xetexko). Tectonic 으로도 빌드 가능 (build.sh 참고).
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pnuthesis}[2026/09/21 v0.1 Pusan National University thesis class (unofficial)]

\newif\if@pnu@doctor
\newif\if@pnu@korean
\DeclareOption{master}{\@pnu@doctorfalse}
\DeclareOption{doctor}{\@pnu@doctortrue}
\DeclareOption{korean}{\@pnu@koreantrue}
\DeclareOption{english}{\@pnu@koreanfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ExecuteOptions{master,korean}
\ProcessOptions\relax

\LoadClass[11pt,a4paper,oneside,onecolumn,openany]{report}

%% ---------------------------------------------------------------- 용지·여백
\RequirePackage[a4paper,left=30mm,right=25mm,top=35mm,bottom=25mm,footskip=15mm,nohead]{geometry}

%% ---------------------------------------------------------------- 글꼴 (XeLaTeX)
\RequirePackage{fontspec}
\RequirePackage{kotex}
%% Tectonic 의 XeTeX 는 ICU 한국어 줄바꿈 자원을 포함하지 않아 오류가 나므로
%% ICU 줄바꿈을 끄고 xetexko 의 자체 글자 간 줄바꿈을 사용한다 (TeX Live 에서도 무해).
\XeTeXlinebreaklocale ""
\IfFileExists{fonts/NanumMyeongjo-Regular.ttf}{%
  \setmainhangulfont{NanumMyeongjo}[Path=fonts/,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-Bold]
  \setsanshangulfont{NanumGothic}[Path=fonts/,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-Bold]
  \setmonohangulfont{NanumGothic}[Path=fonts/,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-Bold]
}{%
  \setmainhangulfont{NanumMyeongjo}[BoldFont={NanumMyeongjo Bold}]
  \setsanshangulfont{NanumGothic}[BoldFont={NanumGothic Bold}]
  \setmonohangulfont{NanumGothic}[BoldFont={NanumGothic Bold}]
}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{texgyrecursor}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic,Scale=MatchLowercase]

%% ---------------------------------------------------------------- 공통 패키지
\RequirePackage{setspace}
\RequirePackage{amsmath,amssymb,amsthm,mathtools,bm}
\RequirePackage{graphicx}
\RequirePackage{booktabs,array,multirow,tabularx,longtable}
\RequirePackage{caption}
\RequirePackage{tocloft}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{xcolor}
\RequirePackage{rotating}
\RequirePackage{float}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage[hidelinks,bookmarksnumbered,pdfencoding=auto,unicode]{hyperref}

%% ---------------------------------------------------------------- 이름
\if@pnu@korean
  \renewcommand{\contentsname}{차\quad 례}
  \renewcommand{\listtablename}{표\ 차례}
  \renewcommand{\listfigurename}{그림\ 차례}
  \renewcommand{\bibname}{참고\ 문헌}
  \renewcommand{\figurename}{그림}
  \renewcommand{\tablename}{표}
  \renewcommand{\appendixname}{부록}
  \floatname{algorithm}{알고리즘}
  \renewcommand{\listalgorithmname}{알고리즘 차례}
\else
  \renewcommand{\bibname}{References}
\fi

%% ---------------------------------------------------------------- 본문 서식
\setlength{\parindent}{15pt}
\setlength{\parskip}{0pt}
\tolerance=1500 \emergencystretch=2em \hfuzz=1pt
\newcommand{\pnu@bodystretch}{1.6}
\captionsetup{labelsep=quad,font=small,labelfont=bf,skip=6pt}
\captionsetup[table]{position=above}
\captionsetup[figure]{position=below}
\renewcommand{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand{\thetable}{\thechapter.\arabic{table}}
\renewcommand{\theequation}{\thechapter.\arabic{equation}}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}

%% 장·절 제목 (고딕, 굵게): 제 1 장 (16 pt) / 1.1 (14 pt) / 1.1.1 (12 pt)
\if@pnu@korean
  \titleformat{\chapter}[block]{\centering\sffamily\bfseries\fontsize{16pt}{20pt}\selectfont}{제\ \thechapter\ 장}{1em}{}
\else
  \titleformat{\chapter}[block]{\centering\sffamily\bfseries\fontsize{16pt}{20pt}\selectfont}{Chapter\ \thechapter}{1em}{}
\fi
\titlespacing*{\chapter}{0pt}{0pt}{28pt}
\titleformat{\section}{\sffamily\bfseries\fontsize{14pt}{18pt}\selectfont}{\thesection}{1em}{}
\titlespacing*{\section}{0pt}{18pt plus 4pt minus 2pt}{8pt}
\titleformat{\subsection}{\sffamily\bfseries\fontsize{12pt}{15pt}\selectfont}{\thesubsection}{1em}{}
\titlespacing*{\subsection}{0pt}{12pt plus 3pt minus 2pt}{5pt}
\titleformat{\subsubsection}{\sffamily\bfseries\normalsize}{\thesubsubsection}{1em}{}
\titlespacing*{\subsubsection}{0pt}{10pt plus 2pt minus 1pt}{3pt}

%% 차례 서식
\renewcommand{\cfttoctitlefont}{\hfill\sffamily\bfseries\fontsize{15pt}{19pt}\selectfont}
\renewcommand{\cftaftertoctitle}{\hfill\mbox{}}
\renewcommand{\cftlottitlefont}{\hfill\sffamily\bfseries\fontsize{15pt}{19pt}\selectfont}
\renewcommand{\cftafterlottitle}{\hfill\mbox{}}
\renewcommand{\cftloftitlefont}{\hfill\sffamily\bfseries\fontsize{15pt}{19pt}\selectfont}
\renewcommand{\cftafterloftitle}{\hfill\mbox{}}
\setlength{\cftbeforetoctitleskip}{0pt}
\setlength{\cftaftertoctitleskip}{24pt}
\setlength{\cftbeforelottitleskip}{0pt}
\setlength{\cftafterlottitleskip}{24pt}
\setlength{\cftbeforeloftitleskip}{0pt}
\setlength{\cftafterloftitleskip}{24pt}
\if@pnu@korean
  \renewcommand{\cftchappresnum}{제\ }
  \renewcommand{\cftchapaftersnum}{\ 장}
  \setlength{\cftchapnumwidth}{5.2em}
\fi
\renewcommand{\cftchapfont}{\sffamily\bfseries}
\renewcommand{\cftchappagefont}{\bfseries}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\setlength{\cftbeforechapskip}{8pt}
\setlength{\cftsecindent}{1.5em}
\setlength{\cftsecnumwidth}{2.6em}
\setlength{\cftsubsecindent}{4.1em}
\setlength{\cftsubsecnumwidth}{3.4em}
\setlength{\cfttabindent}{0pt}
\setlength{\cftfigindent}{0pt}
\renewcommand{\cfttabpresnum}{\tablename\ }
\renewcommand{\cfttabaftersnum}{}
\setlength{\cfttabnumwidth}{4.2em}
\renewcommand{\cftfigpresnum}{\figurename\ }
\setlength{\cftfignumwidth}{4.8em}

%% 정리 환경 (정의는 반드시 출처를 함께 적는다)
\theoremstyle{definition}
\newtheorem{definition}{정의}[chapter]
\newtheorem{assumption}{가정}[chapter]
\newtheorem{proposition}{명제}[chapter]
\newtheorem{remark}{참고}[chapter]
\newtheorem{example}{예}[chapter]

%% ---------------------------------------------------------------- 메타 정보
\newcommand{\pnu@undefined}[1]{\ClassError{pnuthesis}{#1 이(가) 지정되지 않았습니다 (meta.tex 확인)}{}}
\newcommand{\pnutitle}[1]{\gdef\@pnutitle{#1}}            \def\@pnutitle{\pnu@undefined{\string\pnutitle}}
\newcommand{\pnutitleen}[1]{\gdef\@pnutitleen{#1}}        \def\@pnutitleen{\pnu@undefined{\string\pnutitleen}}
\newcommand{\pnuauthor}[1]{\gdef\@pnuauthor{#1}}          \def\@pnuauthor{\pnu@undefined{\string\pnuauthor}}
\newcommand{\pnuauthorspaced}[1]{\gdef\@pnuauthorspaced{#1}} \def\@pnuauthorspaced{\@pnuauthor}
\newcommand{\pnuauthoren}[1]{\gdef\@pnuauthoren{#1}}      \def\@pnuauthoren{\pnu@undefined{\string\pnuauthoren}}
\newcommand{\pnuadvisor}[1]{\gdef\@pnuadvisor{#1}}        \def\@pnuadvisor{\pnu@undefined{\string\pnuadvisor}}
\newcommand{\pnuadvisoren}[1]{\gdef\@pnuadvisoren{#1}}    \def\@pnuadvisoren{}
\newcommand{\pnudept}[1]{\gdef\@pnudept{#1}}              \def\@pnudept{\pnu@undefined{\string\pnudept}}
\newcommand{\pnudepten}[1]{\gdef\@pnudepten{#1}}          \def\@pnudepten{\pnu@undefined{\string\pnudepten}}
\newcommand{\pnugradschool}[1]{\gdef\@pnugradschool{#1}}  \def\@pnugradschool{대학원}   % 예: 데이터사이언스전문대학원
\newcommand{\pnugradschoolen}[1]{\gdef\@pnugradschoolen{#1}} \def\@pnugradschoolen{The Graduate School}
\newcommand{\pnudegree}[1]{\gdef\@pnudegree{#1}}          \def\@pnudegree{공학}
\newcommand{\pnugraddate}[1]{\gdef\@pnugraddate{#1}}      \def\@pnugraddate{\pnu@undefined{\string\pnugraddate}}
\newcommand{\pnuapprovaldate}[1]{\gdef\@pnuapprovaldate{#1}} \def\@pnuapprovaldate{\pnu@undefined{\string\pnuapprovaldate}}
\newcommand{\pnucommittee}[3]{\gdef\@pnuchair{#1}\gdef\@pnumemberA{#2}\gdef\@pnumemberB{#3}}
\def\@pnuchair{}\def\@pnumemberA{}\def\@pnumemberB{}
\newcommand{\pnukeywords}[1]{\gdef\@pnukeywords{#1}}      \def\@pnukeywords{}
\newcommand{\pnukeywordsen}[1]{\gdef\@pnukeywordsen{#1}}  \def\@pnukeywordsen{}

\if@pnu@doctor
  \def\pnu@degreelabel{박\ 사\ 학\ 위\ 논\ 문}
  \def\pnu@degreeshort{박사}
\else
  \def\pnu@degreelabel{석\ 사\ 학\ 위\ 논\ 문}
  \def\pnu@degreeshort{석사}
\fi

%% ---------------------------------------------------------------- 1구역: 표지
\newcommand{\makecover}{%
  \pagenumbering{Alph}% 표지·면지·인준지: 번호 없음 (hyperref 앵커 충돌 방지용 내부 번호)
  \thispagestyle{empty}%
  \begin{center}
    \sffamily
    {\fontsize{16pt}{20pt}\selectfont \pnu@degreelabel\par}
    \vspace{3.2cm}
    {\setstretch{1.3}\fontsize{22pt}{28pt}\selectfont\bfseries \@pnutitle\par}
    \vfill
    {\fontsize{16pt}{20pt}\selectfont \@pnuauthorspaced\par}
    \vfill
    {\fontsize{16pt}{20pt}\selectfont 부산대학교\ \@pnugradschool\par}
    \vspace{0.6em}
    {\fontsize{16pt}{20pt}\selectfont \@pnudept\par}
    \vspace{2.2cm}
    {\fontsize{16pt}{20pt}\selectfont \@pnugraddate\par}
    \vspace*{0.4cm}
  \end{center}
  \clearpage
}

%% 등표지(책등): 제본소에서 만드는 경우가 많으므로 선택 사용
\newcommand{\makespine}{%
  \thispagestyle{empty}%
  \noindent\begin{turn}{-90}%
    \begin{minipage}{0.9\paperheight}
      \sffamily\fontsize{12pt}{14pt}\selectfont
      \pnu@degreeshort\ \ \ \@pnutitle\hfill\@pnuauthorspaced\hfill\@pnugraddate
    \end{minipage}%
  \end{turn}
  \clearpage
}

%% 면지 (표지 다음 백지)
\newcommand{\makeblankpage}{\thispagestyle{empty}\mbox{}\clearpage}

%% ---------------------------------------------------------------- 속표지 및 인준지
\newcommand{\makeapproval}{%
  \thispagestyle{empty}%
  \begin{center}
    \sffamily
    {\setstretch{1.3}\fontsize{22pt}{28pt}\selectfont\bfseries \@pnutitle\par}
    \vspace{1.6cm}
    {\fontsize{16pt}{20pt}\selectfont 이\ 논문을\ \@pnudegree\pnu@degreeshort\ 학위논문으로\ 제출함\par}
    \vspace{1.3cm}
    {\setstretch{1.5}\fontsize{14pt}{18pt}\selectfont
      \@pnuauthorspaced\\
      부산대학교\ \@pnugradschool\\
      \@pnudept\\[0.4em]
      지\ 도\ 교\ 수\quad \@pnuadvisor\par}
    \vfill
    {\fontsize{16pt}{20pt}\selectfont \@pnuauthor 의\ \@pnudegree\pnu@degreeshort\ 학위논문을\ 인준함\par}
    \vspace{0.9cm}
    {\fontsize{14pt}{18pt}\selectfont \@pnuapprovaldate\par}
    \vspace{1.0cm}
    {\setstretch{2.0}\fontsize{14pt}{18pt}\selectfont
      \begin{tabular}{@{}l@{\quad}c@{\ }l@{}}
        위원장 & \underline{\makebox[5.5cm]{\@pnuchair}}   & 인\\
        위\ \ 원 & \underline{\makebox[5.5cm]{\@pnumemberA}} & 인\\
        위\ \ 원 & \underline{\makebox[5.5cm]{\@pnumemberB}} & 인\\
      \end{tabular}\par}
    \vspace*{0.8cm}
  \end{center}
  \clearpage
}

%% ---------------------------------------------------------------- 2구역: 차례
\newcommand{\makefrontmatter}{%
  \pagenumbering{roman}
  \pagestyle{plain}
  {\setstretch{1.3}\tableofcontents}
  \clearpage
  {\setstretch{1.3}\listoftables}
  \clearpage
  {\setstretch{1.3}\listoffigures}
  \clearpage
}

%% 초록 (국문) — 2구역 마지막
\newenvironment{abstractko}{%
  \clearpage
  \thispagestyle{plain}
  \begin{center}
    {\setstretch{1.3}\sffamily\bfseries\fontsize{14pt}{18pt}\selectfont \@pnutitle\par}
    \vspace{1.0cm}
    {\fontsize{12pt}{15pt}\selectfont \@pnuauthorspaced\par}
    \vspace{0.5cm}
    {\fontsize{11pt}{14pt}\selectfont 부산대학교\ \@pnugradschool\ \@pnudept\par}
    \vspace{1.2cm}
    {\sffamily\bfseries\fontsize{12pt}{15pt}\selectfont 요\quad 약\par}
    \vspace{0.6cm}
  \end{center}
  \setstretch{1.5}\fontsize{11pt}{14pt}\selectfont
}{%
  \ifx\@pnukeywords\@empty\else
    \par\vspace{1.2em}\noindent\textbf{주요어:}\ \@pnukeywords
  \fi
  \clearpage
}

%% 초록 (영문) — 논문 맨 끝
\newenvironment{abstracten}{%
  \clearpage
  \thispagestyle{plain}
  \begin{center}
    {\setstretch{1.3}\sffamily\bfseries\fontsize{14pt}{18pt}\selectfont \@pnutitleen\par}
    \vspace{1.0cm}
    {\fontsize{12pt}{15pt}\selectfont \@pnuauthoren\par}
    \vspace{0.5cm}
    {\fontsize{11pt}{14pt}\selectfont Department of \@pnudepten\\ \@pnugradschoolen\\ Pusan National University\par}
    \vspace{1.2cm}
    {\sffamily\bfseries\fontsize{12pt}{15pt}\selectfont Abstract\par}
    \vspace{0.6cm}
  \end{center}
  \setstretch{1.5}\fontsize{11pt}{14pt}\selectfont
}{%
  \ifx\@pnukeywordsen\@empty\else
    \par\vspace{1.2em}\noindent\textbf{Keywords:}\ \@pnukeywordsen
  \fi
  \clearpage
}

%% ---------------------------------------------------------------- 3구역: 본문 시작
\newcommand{\startmainmatter}{%
  \clearpage
  \pagenumbering{arabic}
  \setcounter{page}{1}
  \pagestyle{plain}
  \setstretch{\pnu@bodystretch}
}

%% 부록: 제 A 장 대신 "부록 A"
\let\pnu@origappendix\appendix
\renewcommand{\appendix}{%
  \pnu@origappendix
  \if@pnu@korean
    \titleformat{\chapter}[block]{\centering\sffamily\bfseries\fontsize{16pt}{20pt}\selectfont}{부록\ \thechapter}{1em}{}
    \renewcommand{\cftchappresnum}{부록\ }
    \renewcommand{\cftchapaftersnum}{}
  \else
    \titleformat{\chapter}[block]{\centering\sffamily\bfseries\fontsize{16pt}{20pt}\selectfont}{Appendix\ \thechapter}{1em}{}
    \renewcommand{\cftchappresnum}{Appendix\ }
  \fi
}

%% 감사의 글 (선택)
\newcommand{\acknowledgements}[1]{%
  \chapter*{감사의\ 글}
  \addcontentsline{toc}{chapter}{감사의 글}
  #1
  \clearpage
}

%% 편의 매크로
\newcommand{\ind}{\mathbb{I}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator{\softmax}{softmax}
\DeclareMathOperator{\softplus}{softplus}
\DeclareMathOperator{\AUC}{AUC}
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}
\endinput
