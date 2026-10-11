# Project continuity

- Before changing this project, read `PRD.md` (especially **Current implementation
  contract — local website**) and `PROGRESS.md`. `PRD.md` is the single source of
  truth. Update it before a material scope/architecture change; record results
  and the next action in `PROGRESS.md`.
- Communicate with the user in Indonesian. Work sequentially and explain what
  changed, what was verified, limitations, and the next user action.
- Current prototype: local browser UI + loopback Python server, pinned SepFormer
  Libri3Mix + ECAPA, MPS with CPU features, 1,500 ms retained history. Default
  filtered hop 200 ms, optional 300 ms comparison, Bypass 20 ms.
- Preserve original recordings, manual enrollments, weights and previous outputs.
  Personal audio, embeddings and web-created profiles belong under the Git-ignored
  `evaluation/data/user_recordings/`. Never retain live conversation PCM by default.
- Keep calibration/startup, model runtime, browser application delay and physical
  audio-route delay separate. Never call file replay a microphone/AirPods test,
  or describe bounded-window SepFormer as a trained causal model.
- The model produces three candidate streams; it does not detect arbitrary speaker
  counts. Automatic enrollment quality and target absence still need listening.
- No dataset/model downloads, cloud audio transfer, publication or deployment are
  implied by local prototype work. Use the existing checkpoints and virtualenv.
- Verification evidence must identify failed timing gates and overloads, even when
  a slower comparison configuration is usable. Read `local_web/VERIFICATION.md`
  for the local website's latest measured status.
