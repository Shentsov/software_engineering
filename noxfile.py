import nox

python_versions = ["3.9"]

@nox.session(python=python_versions)
def lint(session):
    if session.python in ["3.9"]:
        session.install("flake8")
        session.run("flake8", ".")
    else:
        session.log("Skipping linting for Python %s" % session.python)

@nox.session(python=python_versions)
def tests(session):
    if session.python == "3.11":
        session.install("-r", "requirements.txt")
        session.run("pytest", "--version")
        session.log("Skipping tests for Python 3.11")
    else:
        session.install("-r", "requirements.txt")
        session.run("pytest")
