import nox

python_versions = ["3.9"]

@nox.session(python=python_versions)
def lint(session):
    if session.python in ["3.9", "3.10"]:
        session.install("flake8")
        session.run("flake8", ".")
    else:
        session.log("Skipping linting for Python %s" % session.python)

@nox.session(python=python_versions)
def tests(session):
    session.install("-r", "requirements.txt")
    session.run("pytest")
