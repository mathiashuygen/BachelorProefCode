
DIR="cudaReports"

if [ $# -lt 1 ]; then
  FILE_NAME="workload_manager_report.ncu-rep"
  
  if [ -f "$DIR/$FILE_NAME" ]; then
    sudo rm "$DIR/$FILE_NAME"
  fi
  docker build -t workload_manager .
  echo "running docker image"
  docker run --rm --gpus all --user root -it -e REPORT_NAME="$FILE_NAME" -v "$(pwd)/cudaReports:/home/dockeruser/app/reports" workload_manager
else
  SUPPLIED_FILE_NAME="$1"

  if [ -f "$DIR/$SUPPLIED_FILE_NAME" ]; then
    sudo rm "$DIR/$SUPPLIED_FILE_NAME"
  fi
  docker build -t workload_manager .
  docker run --rm --gpus all --user root -it -e REPORT_NAME="$SUPPLIED_FILE_NAME" -v "$(pwd)/cudaReports:/home/dockeruser/app/reports" workload_manager
fi  

