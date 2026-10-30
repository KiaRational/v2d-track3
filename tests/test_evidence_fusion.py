"""CPU regression tests; execute in Slurm via run_evidence_fusion.sbatch --test."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stereo"))
os.environ.setdefault("V2D_ROOT", "/tmp")
from evidence_fusion import (angle, body_pose, centroid_pose, hypotheses, robust_pose,
                             refine, hand_segments, geometry_score, recover_frame)


class EvidenceTests(unittest.TestCase):
    def test_exact_rigid_carry_does_not_spin(self):
        from rigid_carry_refine import lock_interval
        n=30
        H=np.tile(np.eye(4),(n,1,1))
        H[:,:3,:3]=Rotation.from_rotvec(np.c_[np.zeros(n),np.zeros(n),np.linspace(0,.8,n)]).as_matrix()
        W=H.copy()
        W[8:,:3,:3]=W[8:,:3,:3]@Rotation.from_euler("x",120,degrees=True).as_matrix()
        out,G=lock_interval(W,H,5,n)
        local=np.linalg.inv(H[5:])@out[5:]
        self.assertLess(np.max(angle(local[:,:3,:3],G[:3,:3])),1e-10)
        np.testing.assert_allclose(out[:5],W[:5])

    def test_centroid_roundtrip_and_hypotheses(self):
        T = np.eye(4)
        T[:3,:3] = Rotation.from_euler("xyz", [.3,.4,.2]).as_matrix()
        T[:3,3] = [.1,.2,.3]
        c = np.array([.2,-.1,.3])
        np.testing.assert_allclose(body_pose(centroid_pose(T,c),c),T,atol=1e-12)
        hs = hypotheses(T,c,np.eye(3))
        self.assertEqual(len(hs),24)
        for H in hs:
            np.testing.assert_allclose(H[:3,:3]@c+H[:3,3],T[:3,:3]@c+T[:3,3],atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(H[:3,:3]),1.)

    def test_robust_grasp_rejects_half_turn_minority(self):
        T = np.tile(np.eye(4),(20,1,1))
        T[:5,:3,:3] = Rotation.from_euler("x",179,degrees=True).as_matrix()
        T[:5,:3,3] = .1
        out = robust_pose(T,np.ones(20))
        self.assertLess(np.degrees(angle(out[:3,:3],np.eye(3))),1.)
        self.assertLess(np.linalg.norm(out[:3,3]),.002)

    def test_real_hand_turn_preserved_and_false_spin_removed(self):
        n=40
        H=np.tile(np.eye(4),(n,1,1))
        H[:,:3,:3]=Rotation.from_euler("z",np.linspace(0,60,n)[:,None],degrees=True).as_matrix()
        H[:,0,3]=np.linspace(0,.15,n)
        G=np.eye(4); G[:3,3]=[.04,.02,.08]
        truth=H@G
        bad=truth.copy()
        bad[12:27,:3,:3]=bad[12:27,:3,:3]@Rotation.from_euler("x",160,degrees=True).as_matrix()
        observations=[(t,truth[t],1.) for t in list(range(0,12,3))+list(range(27,n,3))]
        seg=[{"side":"right","frames":np.arange(n),"H":H}]
        out,_=refine(bad,observations,seg,np.zeros(n,bool),20,80)
        err=np.degrees(angle(out[:,:3,:3],truth[:,:3,:3]))
        self.assertLess(np.percentile(err,95),5.)
        self.assertGreater(np.degrees(angle(out[0,:3,:3],out[-1,:3,:3])),50.)
        self.assertLess(np.linalg.norm(out[:,:3,3]-truth[:,:3,3],axis=1).max(),.01)

    def test_no_support_does_not_freeze_free_motion(self):
        n=20
        W=np.tile(np.eye(4),(n,1,1)); W[:,0,3]=np.arange(n)*.005
        out,info=refine(W,[(t,W[t],1.) for t in range(n)],[],np.zeros(n,bool),20,30)
        self.assertEqual(info["static_edges"],0)
        np.testing.assert_allclose(out[:,:3,3],W[:,:3,3],atol=.001)

    def test_missing_hands_split_airborne_carry(self):
        n=20
        W=np.tile(np.eye(4),(n,1,1)); W[:,2,3]=.2
        H=W.copy(); H[9:12]=np.nan
        joints=np.repeat(H[:,:3,3][:,None],21,axis=1)
        hands={"right":{"H":H,"J":joints}}
        segs,_=hand_segments(W,np.zeros((10,3)),hands,np.tile(np.eye(4),(n,1,1)),
                            {"up":[0,0,1],"height":0},{"right":list(range(n))},20)
        self.assertEqual(len(segs),2)
        self.assertTrue(all(not set(g["frames"]) & {9,10,11} for g in segs))

    def test_large_pose_recovery_requires_geometry(self):
        rng=np.random.default_rng(8)
        p=rng.uniform([-.06,-.02,-.005],[.06,.02,.005],(1000,3))
        p[:,2]+=.02*(p[:,0]>.015)
        normals=np.tile([0.,0.,-1.],(len(p),1))
        truth=np.eye(4); truth[2,3]=.6
        scan=p+truth[:3,3]
        bad=truth.copy(); bad[:3,:3]=Rotation.from_euler("z",180,degrees=True).as_matrix()
        recovered,info=recover_frame(bad,[truth],p,normals,scan,p.mean(0),np.eye(3))
        self.assertTrue(info["accepted"])
        self.assertLess(geometry_score(recovered,p,normals,scan),geometry_score(bad,p,normals,scan)*.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
