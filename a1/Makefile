FLAGS += -std=c++11

FLAGS += -Wall -Wextra -pedantic -Wno-unused-parameter
FLAGS += -Ofast

FLAGS +=  `pkg-config opencv --cflags --libs`
FLAGS +=  -lboost_system -lboost_program_options -lboost_serialization -lboost_filesystem

assignment1: *.cc *.h
	$(CXX) -o $@ assignment1.cc skinmodel.cc $(FLAGS)

.PHONY : clean
clean:
	rm -f assignment1 graph.txt ROC.png score.txt
