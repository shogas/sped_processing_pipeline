pushd ..\..\Data\Tmp
for /f %%i in ('dir /b /a:d run_*') do (
    rmdir /s /q %%i 2>NUL
)
popd
python compare.py test_run.txt
REM type ..\..\Data\Tmp\run_metadata_* 2>NUL
