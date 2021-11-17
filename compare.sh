BENCHMARK_FILENAME=$1
BENCHMARK_NAME="${BENCHMARK_FILENAME%.*}"
BENCHMARK_REPORT_FOLDER=reports/$BENCHMARK_NAME

if [ -z $2 ]
    then
        GROUPBY_ARG=param
    else
        GROUPBY_ARG=param:$2
fi

pytest-benchmark compare  $BENCHMARK_REPORT_FOLDER/after.json $BENCHMARK_REPORT_FOLDER/before.json --sort="name" --columns=Min,Max,Mean --name=short --group-by=$GROUPBY_ARG