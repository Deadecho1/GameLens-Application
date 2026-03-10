Running the (test) Pipeline:
python main.py \
  --video-dir ./data/videos \
  --event-json-dir ./data/event_jsons \
  --run-json-dir ./data/run_jsons

  Run only event detection:
  python main.py \
  --video-dir ./data/videos \
  --event-json-dir ./data/event_jsons \
  --run-json-dir ./data/run_jsons \
  --only-events

  Run only run export:
  python main.py \
  --video-dir ./data/videos \
  --event-json-dir ./data/event_jsons \
  --run-json-dir ./data/run_jsons \
  --only-export
  
-Videos must be .mp4
-All videos should be placed in the specified video folder
