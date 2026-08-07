## DEVELOPMENT_PROCESS.md


### Issue Management
- All work must be associated with an issue.
- The issue should have an accurate title and description.
- If the issue only affects a single repository, create an issue in that repository.
- If the issues affects multiple repository, create an issue in the RADPS-roadmap repository.
- All issues should either be linked as sub-issues to an appropriate RADPS-roadmap items or included in the RADPS project as independent items. Talk to the RADPS product owner to determine how they would like to track the work.
- Each PR should be linked to an issue.
- Once an issue and PR are closed, a new issue and PR should be created for subsequent work.

### Branching Strategy.

- Each feature and bugfix is implemented in a short-lived branch off main.
- Features should kept as small as possible to maximize CI/CD throughput and avoid rollouts on failure.
- Every quarter, at feature freeze, a stable (release) branch is cut off main and tagged for the release
- The QA testing phase runs on this stable branch, executing the acceptance/integration tests previously defined by stakeholders, while development for the next sprint continues in branches off main.
- Any bug found on the stable branch is fixed on main branch first and added to stable branch so that main remains the source of all commits.
- Release tags are permanent and can be checkout out at any time.
- A defect found in an already-deployed release is fixed in a new feature branch created from the (permanent) release tag.

### Definition of Done for New Features
- Tests written with at least 80% coverage.
- Example notebook created (start from `docs/notebook_template.ipynb`).
- NumPy-style docstrings on all public functions.
- Initial performance testing done using larger datasets and relevant timing comparison with CASA.
