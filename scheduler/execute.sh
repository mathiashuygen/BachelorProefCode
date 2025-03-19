
docker build -t thread_experiment .
docker run --rm --gpus all --user root thread_experiment
  
