if [[ $(basename $PWD) != "brains" ]]; then
    echo "Please run this script from the root of the brains repository"
    exit 1
fi
# check that the current env is brains
if [[ $(conda env list | grep brains | wc -l) -eq 0 ]]; then
    echo "Please activate the brains conda environment"
    exit 1
fi

conda env update --file env.yml
