## Summary

Explain the change in a few sentences. Focus on user-visible behavior, runtime impact, or release relevance.

## Related issue

Link the related issue or explain why this change is standalone.

## Validation

- [ ] `python -m compileall -q src`
- [ ] fast CI-aligned regression suite
- [ ] full test suite, if the change touches runtime behavior or integrations
- [ ] docs updated for user-visible changes

List the exact commands you ran:

```bash
# paste commands here
```

## Risk review

- [ ] touches high-risk actions such as writes, execution, or network behavior
- [ ] changes approval, audit, budget, or policy behavior
- [ ] changes first-run or onboarding expectations
- [ ] changes public docs or release-facing copy

## Notes for reviewers

Add migration notes, tradeoffs, or specific review focus areas if needed.
