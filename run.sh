# Gist: execute these commands to run the optimization code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash csrc/build.sh
pytest -q
python scripts/run_opt.py --N 120 --steps 200
