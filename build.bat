pushd ..\..\data

del run_metadata_*
python ..\code\compare\compare.py test_run.txt
type run_metadata_*

popd
