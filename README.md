# SimpleNet
Implement artificial neural network for MNIST classification task using basic C programming language.

### Purpose

Get my hands dirty and learning deep learning the hard way.

### Special for you

+ For Chinese students-"这是一份可以当作 C 语言大作业的项目"

### Dataset

+ Modified MNIST dataset which can be downloaded from [kaggle](https://www.kaggle.com/c/digit-recognizer).

### Compile and execute instruction

This is a demo project you can modify main.c to customize the program. In windows, you can either compile with `cl main.c simplenet.c simplela.c datareader.c` command or add source files to an visual studio project, in linux, use `gcc` to replace `cl` for compiling.

You will get an executable file named main.exe or some thing like that, run it by click or call in command line as you like.

### Build Status

Completed:

+ Refactoring and saved myself from annoying pointer operations
+ Simple linear algebra structure like vector and metrix, operations like add and multiply
+ Basic net structure with `struct` attribute in C
+ Forward pass in simple feedforward fully connected network
+ Backward pass in simple feedforward fully connected network with mini-batch sized 1

In progress:

+ Backward pass with full batch

Plan:

+ Introduce tools like glog to help with debugging and rewrite test files
+ Write makefile for the project and support compling in Windows and Linux