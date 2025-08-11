.PHONY: setup run-notebooks app test clean

setup:		## create venv & install dependencies
	python -m venv .venv
	.venv\Scripts\activate && pip install -r requirements.txt

run-notebooks:	## run core notebooks headless (optional)
	@echo "Open notebooks/01_*.ipynb in Cursor and run interactively"

app:		## run Streamlit app
	streamlit run src/app.py

test:		## run tests
	pytest -q

clean:		## clean up generated files
	rmdir /s /q .venv 2>nul || echo "No .venv to clean"
	del /q data_proc\*.csv 2>nul || echo "No processed data to clean"

help:		## show this help
	@findstr /B "##" Makefile
