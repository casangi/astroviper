## DEVELOPMENT_PROCESS.md

### Issue Organization

- The RADPS Roadmap Project contains all current and future issues for the Program.
- Issues in the RADPS-roadmap repository will be created to track all feature development. Sub-issues will be created in the relevant individual repositories and linked to the top-level RADPS-roadmap issues.
- Feature development RADPS-roadmap items should take less than one program increment (quarter) to complete and sub-issues should be sized to be one sprint or less. Bugs and maintenance items affecting multiple repositories can be added as top-level RADPS-roadmap issues regardless of length and sub-issues added if appropriate.
- Bugs and maintenance issues affecting a single repository should be created in the relevant repository and added to the RADPS Roadmap Project with the appropriate metadata (at least PI and sprint).

### Issue Management
- All work must be associated with an issue.
- The issue should have an accurate title and description.
- Each PR should be linked to an issue.
- Once an issue and PR are closed, a new issue and PR should be created for subsequent work.

### Branching Strategy

- Each feature and bugfix is implemented in a short-lived (less than a sprint) branch off main.
- Features should be kept as small as possible to maximize CI/CD throughput and avoid rollouts on failure.
- Every quarter, at feature freeze, a stable (release) branch is cut off main and tagged for the release.
- The QA testing phase runs on this stable branch, executing the acceptance/integration tests previously defined by stakeholders, while development for the next sprint continues in branches off main.
- Any bug found on the stable branch is fixed on main branch first and added to stable branch so that main remains the source of all commits.
- Release tags are permanent and can be checked out at any time.
- If it is deemed necessary to fix a defect in an already-deployed release, then it will be fixed in a new feature branch created from the (permanent) release tag.

### Code Review and Pull Requests
- Automated code review will check style, typos, inconsistencies in code, etc.
- Human code review should focus on the overall logic, domain consistency, etc.
- PRs are gates. For a PR to merge, the following must be true:
   - All tests from all workflows should pass.
   - The code coverage report should be green and cannot have missing lines.

### Definition of Done for New Features
- Tests written with at least 80% coverage.
- Example notebook created (start from `docs/notebook_template.ipynb`).
- NumPy-style docstrings on all public functions.
- Initial performance testing done using larger datasets and relevant timing comparison with CASA.
