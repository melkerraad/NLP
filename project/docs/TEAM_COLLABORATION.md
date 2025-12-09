# Team Collaboration Guide

## Communication

### Weekly Meetings
- **Frequency:** Once per week (or as needed)
- **Duration:** 30-60 minutes
- **Format:** 
  - Each member shares progress (5 min each)
  - Discuss blockers and challenges
  - Plan next week's tasks
  - Review and update project plan if needed

### Communication Channels
- [ ] Set up team chat (Slack/Discord/Teams)
- [ ] Use GitHub Issues for task tracking
- [ ] Document decisions in meeting notes

## Git Workflow

### Branching Strategy
- `main` - Production-ready code
- `develop` - Integration branch
- `feature/` - Feature branches (e.g., `feature/data-collection`)
- `fix/` - Bug fixes

### Commit Messages
Use clear, descriptive commit messages:
```
feat: Add course data collection script
fix: Resolve embedding dimension mismatch
docs: Update retrieval system documentation
```

### Pull Request Process
1. Create feature branch from `develop`
2. Make changes and commit
3. Push to remote
4. Create Pull Request
5. Request review from at least one team member
6. Merge after approval

## Task Assignment

### Using GitHub Issues
Create issues for each major task:
- Label by component: `data`, `retrieval`, `generation`, `frontend`
- Assign to team member
- Link to project milestones

### Task Dependencies
Document dependencies between tasks:
- Data collection → Preprocessing → Retrieval setup
- Retrieval → Generation integration
- All components → Frontend integration

## Code Review Guidelines

### What to Review
- Code correctness and logic
- Code style and consistency
- Documentation and comments
- Test coverage
- Performance considerations

### Review Checklist
- [ ] Code follows project style guide
- [ ] Functions are well-documented
- [ ] Error handling is appropriate
- [ ] No hardcoded values (use config/env)
- [ ] Tests pass (if applicable)

## Documentation Standards

### Code Documentation
- Docstrings for all functions/classes
- Inline comments for complex logic
- README in each module directory

### Progress Documentation
- Update `PROJECT_PLAN.md` with completed tasks
- Document decisions in `docs/DECISIONS.md`
- Keep `ARCHITECTURE.md` updated

## File Naming Conventions

### Python Files
- Use snake_case: `data_collection.py`
- Descriptive names: `embed_courses.py` not `embed.py`

### Data Files
- Include date/version: `courses_2025-12-09.json`
- Use descriptive names: `test_queries_v1.json`

### Notebooks
- Use descriptive names: `01_explore_course_data.ipynb`
- Number sequentially for workflow

## Shared Resources

### API Keys
- Store in `.env` file (never commit!)
- Use `python-dotenv` to load
- Document required keys in `README.md`

### Data Files
- Large files: Use Git LFS or external storage
- Small files: Commit to repository
- Document data sources and formats

### Model Files
- Don't commit large model files
- Document where to download models
- Include model version info

## Conflict Resolution

### Code Conflicts
1. Communicate before making conflicting changes
2. Use clear branch names
3. Resolve conflicts promptly
4. Test after merging

### Design Decisions
- Discuss major decisions in team meeting
- Document rationale in `docs/DECISIONS.md`
- Vote if needed (majority rules)

## Meeting Notes Template

```markdown
# Team Meeting - [Date]

## Attendees
- [Name 1]
- [Name 2]
- [Name 3]
- [Name 4]

## Progress Updates
- [Member 1]: [What they accomplished]
- [Member 2]: [What they accomplished]
- ...

## Blockers
- [Issue 1]: [Description and who's handling it]
- ...

## Decisions Made
- [Decision 1]: [Details]
- ...

## Action Items
- [ ] [Task] - Assigned to: [Name] - Due: [Date]
- ...

## Next Meeting
- Date: [Date]
- Time: [Time]
```

## Tips for Success

1. **Communicate Early**: Share blockers immediately
2. **Document Everything**: Future you will thank you
3. **Test Incrementally**: Don't wait until the end
4. **Ask for Help**: Use team expertise
5. **Stay Organized**: Keep files and code organized
6. **Celebrate Wins**: Acknowledge progress and achievements

