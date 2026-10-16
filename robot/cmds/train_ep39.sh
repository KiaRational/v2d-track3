# Full canonical Sharpa FlashSAC recipe (250M steps, 4096 worlds) on our ep39 mini_dust_pan reconstruction.
# scene.support=false: the dust pan only ever rests on the table (= ground plane z=0 after gravity alignment); the
# support-surface step wrote an empty stage, which the scene builder rejects.
# sim.njmax/nconmax 512/256 -> 1024/512 (recon_body values; both 1024/512 and 1536/768 OOM at 4096 worlds on an L40S -> 2048 worlds, same 250M-step budget): the thin flat dust pan lying on the table overflowed the constraint
# buffer (up to 920 needed, dropped contacts); the tissue-box example never overflows.
P=/scratch/kghasemz/v2d/robot/hmd/ego_recon/processed/sequence_id=ep39_dustpan/robot_name=sharpa_wave
python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${P}'" reset.seed=42 logging.mode=disabled scene.support=false sim.njmax=1024 sim.nconmax=512 scene.world_count=2048 \
  output_dir=/scratch/kghasemz/v2d/robot/fc_out/ep39_dustpan
