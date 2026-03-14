$python = ".\.venv_exp\Scripts\python"
$base = "reports/videos"
New-Item -ItemType Directory -Force -Path $base | Out-Null

& $python record_demo.py --mode lstm --sim-speedup 6 --max-sim-time 60 --output "$base/lstm.mp4"
& $python record_demo.py --mode physics --sim-speedup 6 --max-sim-time 60 --output "$base/physics.mp4"
& $python record_demo.py --mode reactive --sim-speedup 6 --max-sim-time 60 --output "$base/reactive.mp4"
