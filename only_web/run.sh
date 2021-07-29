export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

touch log
nohup python -u manage.py runserver 0.0.0.0:8000> log 2>&1 &
tail -f log
