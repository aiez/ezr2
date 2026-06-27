# vim: ts=2 sw=2 sts=2 et :
# knobs only; shared targets live in $(KONFIG)/Makefile
KONFIG ?= ../konfig
APP    := ezr2
MAIN   := test_ezr2.py
EXT    := py
LANG   := python
SRC    := *.py
LINT   := ruff check ezr2.py test_ezr2.py
TOOLS  := python3:run ruff:lint
PKG    := python3 gawk ruff neovim tmux
Font   := 4.7        # a touch smaller than konfig's 5, so ezr2.py fits 6 cols

$(KONFIG)/Makefile:
	@test -f $@ || { echo "missing konfig: git clone https://github.com/aiez/konfig $(KONFIG)"; exit 1; }
include $(KONFIG)/Makefile

# ---- test lanes + studies (repo-specific; after the include) ----
DATA ?= ../optimiz

test:    ## run every self-test (resets seed each)
	@python3 -B test_ezr2.py all

HOLD := $(HOME)/tmp/konfig/ezr2_holdouts.log
$(HOLD): ## holdouts landscape-vs-random over all $(DATA), percentiles
	@mkdir -p $(@D)
	@ls $(DATA)/*.csv | (gshuf 2>/dev/null || sort -R) | \
	 xargs -P 12 -I{} python3 -B -u test_ezr2.py holdouts --file={} 2>/dev/null \
	 | tee $@
	@python3 -B pctl.py < $@

PURE := $(HOME)/tmp/konfig/ezr2_pure.log
$(PURE): ## pure search land-vs-random (no tree) over all $(DATA)
	@mkdir -p $(@D)
	@ls $(DATA)/*.csv | (gshuf 2>/dev/null || sort -R) | \
	 xargs -P 12 -I{} python3 -B -u test_ezr2.py pure --file={} 2>/dev/null \
	 | tee $@
	@python3 -B pctl.py < $@

grow: ## keepf/grow sensitivity study -> Growing.png + Growing.md grid
	@python3 -B Growing.py
