cd ~
DIR="cudaReports"
FILE="matrix_add_report.ncu-rep"

if [ -f "$DIR/$FILE" ]; then
    sudo rm cudaReports/"$FILE"
fi
cd ownCudaPrograms/matrixAdd
docker build -t matrix_adder .
docker run --rm --gpus all --user root -e REPORT_NAME="$FILE" -v ~/cudaReports:/home/dockeruser/app/reports matrix_adder
