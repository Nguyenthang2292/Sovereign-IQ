# Contributing to ATC Serverless

Thank you for your interest in contributing to ATC Serverless! This document outlines the contribution process and guidelines.

## Getting Started

1. **Fork the Repository**: Create your own fork of the repository
2. **Clone Your Fork**: `git clone https://github.com/your-username/atc-serverless.git`
3. **Create a Branch**: `git checkout -b feature/your-feature-name`
4. **Make Changes**: Implement your changes
5. **Test Your Changes**: Run the test suite
6. **Commit Your Changes**: `git commit -m "feat: add your feature"`
7. **Push to Your Fork**: `git push origin feature/your-feature-name`
8. **Create a Pull Request**: Submit a PR against the main repository

## Development Guidelines

### Code Style
- Follow Rust coding standards
- Use `cargo fmt` to format code
- Run `cargo clippy` for linting
- Ensure all tests pass

### Testing
- Add unit tests for new functionality
- Run `cargo test` to verify tests pass
- Consider adding property-based tests with `proptest`

### Documentation
- Update relevant documentation
- Add examples where appropriate
- Ensure API documentation is complete

## Changelog Process

This project uses [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format for version history.

### How to Update the Changelog

1. **For New Features**: Add an entry under "Added" with the version number
2. **For Changes**: Add an entry under "Changed" 
3. **For Bug Fixes**: Add an entry under "Fixed"
4. **For Removals**: Add an entry under "Removed"

### Example Entry
```markdown
## [1.2.0] - 2023-01-15

### Added
- New feature description

### Changed
- Updated existing functionality

### Fixed
- Fixed bug description
```

### Automation
For automated changelog generation, we use `git-cliff`. To generate a changelog:

```bash
# Install git-cliff if not already installed
cargo install git-cliff

# Generate changelog
git-cliff --output CHANGELOG.md
```

### Release Process
1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md` with new version entry
3. Commit changes: `git commit -m "release: version 1.2.0"`
4. Tag the release: `git tag v1.2.0`
5. Push changes and tag: `git push origin main --tags`

## Issue Tracking

- Use GitHub Issues for bug reports and feature requests
- Label issues appropriately (bug, enhancement, documentation, etc.)
- Assign issues to yourself when working on them

## Pull Request Guidelines

- Provide a clear description of your changes
- Reference any related issues
- Ensure tests pass
- Follow the existing code style
- Update documentation as needed

## Support

If you have questions or need help, please:
- Check the existing documentation
- Search GitHub Issues for similar problems
- Create a new issue if your question isn't answered

Happy coding!