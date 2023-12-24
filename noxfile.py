import nox

@nox.session(python=["3.9", "3.10", "3.11"])
def lint(session):
    session.install("flake8")
    session.run("flake8", ".")

@nox.session(python=["3.9", "3.10", "3.11"])
def tests(session):
    session.install("-r", "requiremets.txt")
    session.run("pytest")
