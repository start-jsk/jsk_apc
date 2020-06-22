all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

install: install3

install2:
	PYTHONVERSION=2 .make/install.sh

install3:
	PYTHONVERSION=3 .make/install.sh

lint:
	.make/lint.sh
