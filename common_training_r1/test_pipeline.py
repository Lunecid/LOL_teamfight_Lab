import copy,unittest
import joblib,numpy as np,torch
from common import *
from data import Adapter,snapshots,canonical_order
from models import *

class Gates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.d=joblib.load(HERE/'DATA.joblib');cls.x=cls.d['x'];cls.meta=cls.d['meta'];cls.a=cls.d['adapter'];cls.fit=cls.meta['macro_role']=='FIT'
    def test_input_information_roundtrip(self):
        shapes={k:v.shape[1:] for k,v in self.x.items()};back=unflat(flat(self.x),shapes)
        for k in self.x:np.testing.assert_array_equal(self.x[k],back[k])
        self.assertEqual(set(self.x),{'player','team','rift','relation','present'})
    def test_roles_and_fit_only(self):
        m=self.meta;self.assertFalse(set(m['match_id'][self.fit])&set(m['match_id'][~self.fit]));self.assertEqual(set(m['macro_role']),{'FIT','STOP'})
        with self.assertRaises(AssertionError):Adapter().fit([self.d['observed'][0]],['STOP_15.20'])
        before=stable(repr(self.a.stats));rows=copy.deepcopy(self.d['observed'][:2]);rows[0]['player'][...]=1e30;self.a.transform(rows,3);self.assertEqual(before,stable(repr(self.a.stats)))
    def test_history_lengths_same_current(self):
        rows=self.d['observed'][:10]
        one=self.a.transform(rows,1);five=self.a.transform(rows,5)
        for k in one:np.testing.assert_array_equal(one[k][:,-1],five[k][:,-1])
        self.assertEqual(len(one['present']),len(five['present']))
        for k in ('player','team','rift','relation'):self.assertTrue((five[k][five['present']==0]==0).all())
    def test_prefix_future_mutation_and_boundary(self):
        r=read(HERE/'DB_SELECTION.json')['rows'][0];detail=read(r['detail_path']);tl=read(r['timeline_path']);t=r['t0_ms'];part=detail['info']['participants']
        before=snapshots(tl,part,[t])[t];changed=copy.deepcopy(tl)
        for f in changed['info']['frames']:
            f['events']=[e for e in f.get('events',[]) if e.get('timestamp',0)<=t]
        changed['info']['frames'][-1]['events'] += [{'type':'CHAMPION_KILL','timestamp':t+1,'killerId':1,'victimId':6,'assistingParticipantIds':[]},{'type':'ITEM_UNDO','timestamp':t+2,'participantId':1}]
        after=snapshots(changed,part,[t])[t]
        for key in ('personal','team_extra','rift','item_ok'):np.testing.assert_array_equal(before[key],after[key])
        mini={'info':{'frames':[{'events':[{'type':'CHAMPION_KILL','timestamp':100,'killerId':1,'victimId':6,'assistingParticipantIds':[]}]}]}}
        s=snapshots(mini,part,[99,100]);self.assertEqual(s[99]['personal'][0,0],0);self.assertEqual(s[100]['personal'][0,0],1)
    def test_unknown_onset_not_backward(self):
        p=[{'participantId':i,'teamId':100 if i<=5 else 200} for i in range(1,11)]
        tl={'info':{'frames':[{'events':[{'timestamp':100,'type':'ITEM_UNDO','participantId':1},{'timestamp':200,'type':'CHAMPION_KILL','killerId':1,'victimId':6}]}]}}
        s=snapshots(tl,p,[0,100,200]);self.assertTrue(s[0]['item_ok'][0]);self.assertFalse(s[100]['item_ok'][0]);self.assertTrue(np.isfinite(s[100]['personal'][:,2]).all());self.assertTrue(np.isnan(s[200]['personal'][:,2]).all())
    def test_rift_names_and_unknown_onset(self):
        p=[{'participantId':i,'teamId':100 if i<=5 else 200} for i in range(1,11)]
        tl={'info':{'frames':[{'events':[{'timestamp':100,'type':'DRAGON_SOUL_GIVEN','teamId':0,'name':'Hextech'},{'timestamp':200,'type':'DRAGON_SOUL_GIVEN','teamId':0,'name':'FUTURE_UNKNOWN'}]}]}}
        s=snapshots(tl,p,[0,100,200]);self.assertEqual(s[100]['rift'][7],1);self.assertEqual(s[0]['rift'][7],0);self.assertTrue(np.isnan(s[200]['rift'][3:]).all());self.assertEqual(s[100]['team_extra'][:,-1].sum(),0)
    def test_minibatch_weight_expectation(self):
        mask=np.ones((7,1),bool);m=np.array(['a','a','a','b','c','d','d']);p=np.array(['x','x','x','x','y','y','y']);w=weights(mask,m,p)[:,0];loss=np.arange(7.)**2
        pieces=[np.array([0,3,6]),np.array([1,4]),np.array([2,5])]
        estimated=sum(len(ix)/7*(7/len(ix))*np.sum(w[ix]*loss[ix]) for ix in pieces)
        self.assertAlmostEqual(estimated,np.sum(w*loss))
    def test_canonical_roster_permutation(self):
        r=read(HERE/'DB_SELECTION.json')['rows'][0];d=read(r['detail_path']);tl=read(r['timeline_path']);p=d['info']['participants'];a,t=canonical_order(tl['info']['frames'],p);b,u=canonical_order(tl['info']['frames'],p[::-1]);np.testing.assert_array_equal(a,b);self.assertEqual(t,u);self.assertLessEqual(t,r['t0_ms'])
        anchors=read(HERE/'CANONICAL_ANCHORS.json');self.assertTrue(all(v['anchor_ms']<=90000 for v in anchors.values()))
    def test_final_outcome_fields_ignored(self):
        r=read(HERE/'DB_SELECTION.json')['rows'][0];d=read(r['detail_path']);tl=read(r['timeline_path']);p=d['info']['participants'];poison=copy.deepcopy(p)
        for v in poison:v.update(win=True,kills=999999,deaths=999999,teamPosition='FUTURE_ROLE',item0=99999)
        a=snapshots(tl,p,[r['t0_ms']])[r['t0_ms']];b=snapshots(tl,poison,[r['t0_ms']])[r['t0_ms']]
        for k in ('personal','team_extra','rift','item_ok'):np.testing.assert_array_equal(a[k],b[k])
        np.testing.assert_array_equal(canonical_order(tl['info']['frames'],p)[0],canonical_order(tl['info']['frames'],poison)[0])
    def test_team_sums_and_spatial_activity(self):
        for r in self.d['observed'][:50]:
            for j,valid in enumerate(r['present']):
                if not valid:continue
                p=r['player'][j]
                for team in range(2):
                    v=p[team*5:(team+1)*5];matrix=np.c_[v[:,3],v[:,1],v[:,0],v[:,18:21],v[:,4:6]]
                    expected=np.where(np.isfinite(matrix).all(0),np.nansum(matrix,axis=0),np.nan)
                    np.testing.assert_allclose(r['team'][j,team,:8],expected)
                active=np.isfinite(p[:,16:18]).all(1)&np.isfinite(p[:,8])&(p[:,8]>0)
                np.testing.assert_array_equal(r['relation'][j,:,:,5],active[:,None]&active[None,:]);self.assertEqual(p.shape[0],10)
    def test_weighting_exact_and_empty(self):
        m=np.array(['a','a','b','c']);p=np.array(['x','x','x','y']);mask=np.array([[1,0],[1,0],[1,0],[1,0]],bool)
        w=weights(mask,m,p);np.testing.assert_allclose(w[:,0],[.125,.125,.25,.5]);self.assertEqual(w[:,1].sum(),0)
    def test_masked_loss_poison_and_nb_gradient(self):
        yy=subset(self.d['y'],self.fit);mm=subset(self.meta,self.fit);y,mask,w,spec=setup_targets(yy,mm)
        ix=np.arange(min(64,len(y['gold'])));ym={k:torch.from_numpy(v[ix]).float() for k,v in y.items()};ma={k:torch.from_numpy(v[ix]) for k,v in mask.items()};ww={k:torch.from_numpy(v[ix]).float() for k,v in w.items()}
        # Explicit unsupported target, including a different group's supported peers.
        ma['gold'][0,0]=False;ww['gold'][0,0]=0
        pred={'gold':torch.zeros(len(ix),6,requires_grad=True),'count':torch.zeros(len(ix),84,requires_grad=True),'hazard':torch.zeros(len(ix),3,3,requires_grad=True)}
        original=torch_loss(pred,ym,ma,ww,spec);poison={k:v.clone() for k,v in ym.items()}
        for k in poison:poison[k][~ma[k]]=float('nan')
        result=torch_loss(pred,poison,ma,ww,spec);self.assertTrue(torch.isfinite(result));self.assertAlmostEqual(original.item(),result.item(),places=6);result.backward();self.assertEqual(pred['gold'].grad[0,0],0)
        x=.7;v=3.;r=2.;eps=1e-4;g,h=nb_derivatives(x,v,r);fd=(nb_nll(v,x+eps,r)-nb_nll(v,x-eps,r))/(2*eps);fh=(nb_nll(v,x+eps,r)-2*nb_nll(v,x,r)+nb_nll(v,x-eps,r))/eps**2;self.assertAlmostEqual(g,fd,places=5);self.assertAlmostEqual(h,fh,places=5)
    def test_neural_player_permutation_and_team_mixture(self):
        yy=subset(self.d['y'],self.fit);_,_,_,spec=setup_targets(yy,subset(self.meta,self.fit));x=subset(self.x,slice(0,8));net=Hierarchy({k:v.shape[1:] for k,v in x.items()},spec)
        torch.manual_seed(4);nn.init.normal_(net.output.weight,0,.02) # nonconstant predictor
        p=np.array([2,4,0,3,1,9,8,7,6,5]);xp={**x,'player':x['player'][:,:,p],'relation':x['relation'][:,:,p][:,:,:,p]}
        a=neural_predict(net,x);b=neural_predict(net,xp)
        for k in a:np.testing.assert_allclose(a[k],b[k],atol=2e-6,rtol=2e-6)
        b=neural_predict(net,swap(x));s=derived([a,b],spec);ss=derived([b,a],spec)
        np.testing.assert_allclose(s['net_gold'],-ss['net_gold'],atol=1e-5);np.testing.assert_allclose(s['terminal'][:,:,[0,2,1]],ss['terminal'],atol=1e-6)
        self.assertTrue((np.diff(s['terminal'][:,:,1:],axis=1)>=-1e-7).all());np.testing.assert_allclose(s['terminal'].sum(-1),1,atol=1e-6)
        self.assertTrue((np.diff(s['count_positive'],axis=1)>=-1e-7).all())

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Gates);result=unittest.TextTestRunner(verbosity=2).run(suite)
    dump(HERE/'TEST_RESULTS.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)})
    raise SystemExit(0 if result.wasSuccessful() else 1)
