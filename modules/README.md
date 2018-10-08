# Visual Speech Prediction  

### Quick setup:
- `git clone https://gitlab.com/sziraqui/vsp.git`
- `cd vsp`
- **Create python3 environment** [call it `vsp-cpu` or `vsp-gpu`]
- **Activate the environment**
- `make init`
- `make deps-ubuntu` [optional][To install dependencies on Ubuntu 16+ which includes all dependencies for opencv and dlib]
- `make deps-python` [To install required pip packages in current environment]
- `make dlib`   [To install compile and install dlib v19.5 in current python environment]
- `make opencv` [buggy][Use python-opencv from pip if this doesn't work for you or compile yourself and suggest changes to the Makefile]

### Testing the setup
- `cd tests`
- `python3 *.py -v`
- Last line must say 'Test passed' for every .py file

### Important notes:
- **Activate python environment before doing anything**
- Ensure all tests pass before raising pushing your code to upstream repo
- Add necessary tests for every method/class you make

#WeCanMakeItHappen