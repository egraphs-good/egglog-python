
all: python/egglog/visualizer.js



# download visualizer release from github
#
visualizer.tgz:
	curl -s https://api.github.com/repos/egraphs-good/egraph-visualizer/releases/latest \
	| grep "browser_download_url.*tgz" \
	| cut -d : -f 2,3 \
	| tr -d \" \
	| wget -qi - -O visualizer.tgz

# extract visualizer release
python/egglog/visualizer.js python/egglog/visualizer.css: visualizer.tgz
	tar -xzf visualizer.tgz
	rm visualizer.tgz
	mv package/dist/index.js python/egglog/visualizer.js
	mv package/dist/style.css python/egglog/visualizer.css
	rm -rf package

clean:
	rm -rf package python/egglog/visualizer.css python/egglog/visualizer.js visualizer.tgz
