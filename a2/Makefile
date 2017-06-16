FLAGS += -std=c++11

FLAGS += -Wall -Wextra -pedantic
FLAGS += -Ofast
#FLAGS += -D__OPENCV_OBJDETECT_HPP__
FLAGS +=  `pkg-config opencv --cflags --libs`
FLAGS +=  -lboost_system -lboost_program_options -lboost_serialization -lboost_filesystem

assignment2: *.cc *.h
	$(CXX) -o $@ assignment2.cc hog.cc $(FLAGS)

.PHONY : clean
clean:
	rm -f assignment2 graph.txt ROC.png score.txt
