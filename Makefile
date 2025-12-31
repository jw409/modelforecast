.PHONY: test publish arena

# Run local server for arena testing
test:
	@echo "Starting arena on http://localhost:8877/corewars/"
	@cd docs && python3 -m http.server 8877

# Alias for test
arena: test

# Publish changes to GitHub Pages
publish:
	git add -A
	git commit -m "fix: arena updates" || true
	git push
	@echo "Pushed to GitHub. Pages will update in ~1 min."
	@echo "View at: https://jw409.github.io/modelforecast/corewars/"
