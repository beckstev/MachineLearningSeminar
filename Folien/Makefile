all: build/Vortrag.pdf

TeXOptions = -lualatex \
				 -interaction=nonstopmode \
				 -halt-on-error \
				 -output-directory=build

build/Vortrag.pdf: FORCE | build
	latexmk $(TexOptions) Vortrag.tex

light: Vortrag.tex
	lualatex -interaction=nonstopmode \
	-halt-on-error \
	-output-directory=build \
	Vortrag.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
