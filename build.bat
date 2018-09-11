pushd ..\..\Data\Tmp
for /f %%i in ('dir /b /a:d run_*') do (
    rmdir /s /q %%i 2>NUL
)
popd
python factorize.py test_run_sample.txt
REM python result_compare.py ..\..\Data\Tmp\compare_test
REM type ..\..\Data\Tmp\run_metadata_* 2>NUL
