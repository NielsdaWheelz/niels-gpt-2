so the policy is straightforward:
	•	amp: on by default (mps only)
	•	activation checkpointing: off by default; turn on only when you hit oom
	•	when you turn on checkpointing, immediately “spend” the saved memory by increasing the real objective knobs: T, C/L, or effective batch.
