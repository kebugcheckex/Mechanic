CXX = g++

CFLAGS = -I. -Wno-c++11-narrowing -Wno-unused-parameter -Wunused-function -std=c++11 `pkg-config --cflags opencv tesseract`

LIBS = `pkg-config --libs-only-L --libs-only-l opencv tesseract`

DEPS = group_classifier.h max_meaningful_clustering.h mser.h min_bounding_box.h region_classifier.h region.h text_extract.h utils.h FaceDetector.h TextDetector.h VideoReader.h ImageUtils.h

OBJ = fast_clustering.o group_classifier.o max_meaningful_clustering.o min_bounding_box.o mser.o nfa.o utils.o region_classifier.o region.o FaceDetector.o TextDetector.o VideoReader.o ImageUtils.o main.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

all: Mechenic

Mechanic: $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -rf $(OBJ) Mechanic
