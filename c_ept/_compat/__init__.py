"""Packaged legacy module files, found through c_ept.__path__.

They belong to the c_ept namespace. Hide the storage namespace from recursive
pkgutil discovery so it does not advertise a second set of legacy aliases.
"""
__path__ = []
