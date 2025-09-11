# Placeholder Audit - Documentation Updates Needed

## 🎯 **Summary**
Found multiple placeholders, outdated URLs, and placeholder content that need to be updated now that we have the actual domain `pihk.ai` and project details.

---

## 📋 **Critical Updates Needed**

### **1. Package Configuration (`pyproject.toml`)**
**Current (Placeholder):**
```toml
authors = [
    {name = "Neuromod Research Team", email = "research@neuromod.ai"}
]
Homepage = "https://github.com/neuromod-ai/neuromod-llm"
Documentation = "https://neuromod-llm.readthedocs.io"
Repository = "https://github.com/neuromod-ai/neuromod-llm.git"
Issues = "https://github.com/neuromod-ai/neuromod-llm/issues"
```

**Should be:**
```toml
authors = [
    {name = "Cris Neckar", email = "cris@tbc-cris.local"}
]
Homepage = "https://pihk.ai"
Documentation = "https://pihk.ai"
Repository = "https://github.com/cneckar/neuromod-llm-poc.git"
Issues = "https://github.com/cneckar/neuromod-llm-poc/issues"
```

### **2. Website Links (`docs/website/index.html`)**
**Current (Placeholder):**
- Multiple `href="#"` links in footer
- Missing actual documentation links
- Placeholder social media links

**Should be:**
- Link to actual GitHub repository
- Link to actual documentation
- Remove or update placeholder social links

### **3. Documentation URLs**
**Files with outdated URLs:**
- `examples/01_quick_start.py` - `https://neuromod-llm.readthedocs.io`
- `INSTALLATION.md` - `https://neuromod-llm.readthedocs.io`

**Should be:**
- `https://pihk.ai` (main site)
- `https://github.com/cneckar/neuromod-llm-poc` (repository)

### **4. Analysis Files**
**Files with placeholder URLs:**
- `analysis/data_code_release.py` - `https://github.com/your-org/neuromod-llm-poc`
- `analysis/data_code_release.py` - `email: ANONYMIZED@example.com`

**Should be:**
- `https://github.com/cneckar/neuromod-llm-poc`
- Actual contact email or remove

### **5. API Documentation**
**Files with placeholder endpoints:**
- `vertex_container/API_EXAMPLES.md` - `https://your-endpoint/predict`
- `api/README.md` - `https://your-endpoint.vertex.ai`

**Should be:**
- Update with actual deployment URLs or mark as examples

---

## 🔧 **Detailed Fix List**

### **High Priority (Critical for Launch)**

1. **`pyproject.toml`** - Update package metadata
2. **`docs/website/index.html`** - Fix footer links
3. **`examples/01_quick_start.py`** - Update documentation URLs
4. **`INSTALLATION.md`** - Update documentation URLs

### **Medium Priority (Important for Users)**

5. **`analysis/data_code_release.py`** - Update repository URLs
6. **`vertex_container/API_EXAMPLES.md`** - Update endpoint examples
7. **`api/README.md`** - Update endpoint examples

### **Low Priority (Nice to Have)**

8. **`vertex_container/README.md`** - Update project ID placeholders
9. **`api/server.py`** - Update project ID placeholders
10. **Various test files** - Update placeholder URLs

---

## 📝 **Specific Changes Needed**

### **1. pyproject.toml**
```toml
# Change from:
authors = [
    {name = "Neuromod Research Team", email = "research@neuromod.ai"}
]
Homepage = "https://github.com/neuromod-ai/neuromod-llm"
Documentation = "https://neuromod-llm.readthedocs.io"
Repository = "https://github.com/neuromod-ai/neuromod-llm.git"
Issues = "https://github.com/neuromod-ai/neuromod-llm/issues"

# Change to:
authors = [
    {name = "Cris Neckar", email = "cris@tbc-cris.local"}
]
Homepage = "https://pihk.ai"
Documentation = "https://pihk.ai"
Repository = "https://github.com/cneckar/neuromod-llm-poc.git"
Issues = "https://github.com/cneckar/neuromod-llm-poc/issues"
```

### **2. docs/website/index.html**
```html
<!-- Change from: -->
<a href="#" class="btn btn-primary">Read Docs</a>
<a href="#" class="btn btn-primary">Join Discussion</a>

<!-- Change to: -->
<a href="https://github.com/cneckar/neuromod-llm-poc" class="btn btn-primary">Read Docs</a>
<a href="https://github.com/cneckar/neuromod-llm-poc/discussions" class="btn btn-primary">Join Discussion</a>
```

### **3. examples/01_quick_start.py**
```python
# Change from:
print("2. Read the Documentation: Visit https://neuromod-llm.readthedocs.io")
print("- Documentation: https://neuromod-llm.readthedocs.io")

# Change to:
print("2. Read the Documentation: Visit https://pihk.ai")
print("- Documentation: https://pihk.ai")
```

### **4. INSTALLATION.md**
```markdown
<!-- Change from: -->
- **Documentation**: https://neuromod-llm.readthedocs.io

<!-- Change to: -->
- **Documentation**: https://pihk.ai
```

### **5. analysis/data_code_release.py**
```python
# Change from:
repository_url="https://github.com/your-org/neuromod-llm-poc",
documentation_url=f"https://github.com/your-org/neuromod-llm-poc/releases/tag/v{version}",

# Change to:
repository_url="https://github.com/cneckar/neuromod-llm-poc",
documentation_url=f"https://github.com/cneckar/neuromod-llm-poc/releases/tag/v{version}",
```

---

## 🚀 **Implementation Plan**

### **Phase 1: Critical Updates (Do First)**
1. Update `pyproject.toml` with correct metadata
2. Fix website footer links
3. Update documentation URLs in examples and installation guide

### **Phase 2: Secondary Updates**
4. Update analysis files with correct repository URLs
5. Update API documentation with proper examples
6. Update remaining placeholder URLs

### **Phase 3: Final Cleanup**
7. Review all files for remaining placeholders
8. Test all links work correctly
9. Update any remaining documentation

---

## ✅ **Verification Checklist**

- [ ] All `pyproject.toml` URLs point to correct repository
- [ ] Website footer links work correctly
- [ ] Documentation URLs point to `pihk.ai`
- [ ] Analysis files use correct repository URLs
- [ ] API examples are properly formatted
- [ ] No broken links in documentation
- [ ] All placeholder emails are updated or removed
- [ ] Social media links are updated or removed

---

## 🎯 **Expected Outcome**

After these updates:
- All documentation will have correct URLs
- Package metadata will be accurate
- Website will have working links
- Users will be directed to the right resources
- Project will look professional and complete

This will ensure that when users visit `pihk.ai` or install the package, they get accurate information and working links throughout the entire project.
