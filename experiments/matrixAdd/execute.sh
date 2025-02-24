

DIR="cudaReports"

if [ $# -lt 1 ]; then
  FILE_NAME="matrix_add"

  if [ -f "$DIR/$FILE_NAME" ]; then
    sudo rm "$DIR/$FILE_NAME"
  fi
  docker build -t matrix_add .
  docker run --rm --gpus all --user root -e REPORT_NAME="$FILE_NAME" -v "$(pwd)/cudaReports:/home/dockeruser/app/reports" matrix_add
else
  SUPPLIED_FILE_NAME="$1"

  if [ -f "$DIR/$SUPPLIED_FILE_NAME" ]; then
    sudo rm "$DIR/$SUPPLIED_FILE_NAME"
  fi
  docker build -t matrix_add .
  docker run --rm --gpus all --user root -e REPORT_NAME="$SUPPLIED_FILE_NAME" -v "$(pwd)/cudaReports:/home/dockeruser/app/reports" matrix_add
fi  
