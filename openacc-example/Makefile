
CXX=pgc++
CXXFLAGS=-std=c++11 -ta=nvidia:cuda11.0,managed -fast
CXXFLAGS_LOOP=-Minfo=all,accel,intensity

OBJECTS=main.o gaussian.o noacc_gaussian.o
DOT_OBJECTS=dot_main.o dot.o noacc_dot.o

all: main dot

dot: $(DOT_OBJECTS)
	$(CXX) $(CXXFLAGS) -o dot $(DOT_OBJECTS)

main: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o main $(OBJECTS)

dot.o: dot.h
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_LOOP) -c dot.cc

gaussian.o: gaussian.h point.h
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_LOOP) -c gaussian.cc

# Implicit rules.
main.o: gaussian.h noacc_gaussian.h point.h
noacc_gaussian.o: noacc_gaussian.h gaussian.h point.h
dot_main.o: dot.h noacc_dot.h
noacc_dot.o: noacc_dot.h

.PHONY: clean
clean:
	rm main dot $(OBJECTS) $(DOT_OBJECTS)
