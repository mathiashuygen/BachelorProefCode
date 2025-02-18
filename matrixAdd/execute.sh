cd ~
cd Documents
DIR="~/cudaReports"

if[ -d "$DIR"]; then
  sudo rm -rf cudaReports
fi

cd /ownCudaPrograms/matrixAdd
docker build -t matrix_adder .
docker run --rm --gpus all --user root -v ~/cudaReports:/home/dockeruser/app/reports matrix_adder
