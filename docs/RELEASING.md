# Preparing the archival release

1. Complete the audit gates in AUDIT.md and review CHANGELOG.md. Test real
   science recipes separately from memory-safety tests. Inspect CI results for
   the installed-wheel platform matrix; do not call unrun platforms verified.
2. Confirm the final version, creator list, affiliations and ORCIDs with the
   maintainers. CITATION.cff currently identifies the wrapper maintainer and an
   unreleased candidate; it deliberately contains no DOI or release date.
   Retain Andy Becker's upstream MIT copyright notice in both distributions.
3. Build an sdist and wheel from a clean tagged commit. Verify both license files,
   runtime dependencies and compiled extension. Install the wheel in a clean
   environment and test outside the source tree.
4. Enable Zenodo's GitHub integration before publishing the final GitHub release,
   or upload the exact release archive manually. Check Zenodo's record metadata
   before publishing. Archive the software, not only a short prose notice.
5. Add the assigned version DOI to citation metadata and the README. Cite that
   version in the SynDiff method paper. Do not retag a published version.
6. Submit an ASCL entry with a short description, authors, code/archive links,
   contact details, preferred citation and associated research reference or
   submitted-paper status. Distinguish wrapper contributions from HOTPANTS.

Official references:
- [Zenodo software metadata](https://help.zenodo.org/docs/github/describe-software/)
  (`.zenodo.json`, if added, overrides CITATION.cff metadata).
- [ASCL submission](https://ascl.net/code/submit).
- [NumPy downstream ABI guidance](https://numpy.org/doc/2.0/dev/depending_on_numpy.html).
- [Setuptools license metadata](https://setuptools.pypa.io/en/stable/userguide/license_migration.html).

No step above authorizes publishing a release, DOI record, ASCL submission or
changing the live editable installation. Those are separate deployment actions.
