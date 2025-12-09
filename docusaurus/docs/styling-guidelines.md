# Docusaurus Theme and Styling Guidelines

## Overview

This document provides guidelines for maintaining consistent visual style, responsive design, and accessibility compliance across the Physical AI & Humanoid Robotics textbook. These guidelines ensure a professional, accessible, and user-friendly experience for all learners.

## Visual Style Consistency

### Typography Standards

#### Headings Hierarchy
```markdown
# Module Title (H1) - Page Title
## Section Title (H2) - Major Sections
### Subsection Title (H3) - Content Sections
#### Detail Title (H4) - Subtopics
```

#### Text Formatting
- **Bold text**: For key terms and important concepts
- *Italic text*: For emphasis and definitions
- `Code snippets`: For ROS commands, file names, and technical terms
- [Links](#): For cross-references and external resources

### Color Scheme

#### Primary Colors
- **Text**: `#1c1e21` - Main text color for readability
- **Headers**: `#242526` - Slightly darker for hierarchy
- **Links**: `#2980b9` - Professional blue for links
- **Code blocks**: `#f6f8fa` - Light gray background for code

#### Accent Colors
- **Success**: `#27ae60` - For completed steps and positive feedback
- **Warning**: `#f39c12` - For important notices and warnings
- **Error**: `#e74c3c` - For critical information and errors
- **Info**: `#3498db` - For informational content

### Layout Standards

#### Content Structure
Each document should follow this structure:
1. **Title** - Clear, descriptive heading
2. **Overview** - Brief introduction to the topic
3. **Main Content** - Detailed information with appropriate headings
4. **Examples** - Practical examples and code snippets
5. **Summary** - Key takeaways
6. **Next Steps** - Connection to subsequent topics

#### Spacing and Margins
- Use consistent spacing between sections
- Maintain adequate white space for readability
- Follow Docusaurus default padding and margins

## Responsive Design Implementation

### Mobile-First Approach

#### Content Adaptation
- Ensure all text remains readable on small screens
- Use responsive tables that scroll horizontally if needed
- Optimize images for various screen sizes

#### Navigation Optimization
- Main navigation collapses to hamburger menu on mobile
- Breadcrumb navigation for easy backtracking
- Sticky headers for easy access to main content

### Breakpoint Standards

#### Screen Size Categories
- **Mobile**: Up to 768px width
- **Tablet**: 769px to 1024px width
- **Desktop**: 1025px and above

#### Content Adjustments
```css
/* Example responsive adjustments */
@media (max-width: 768px) {
  .code-block {
    font-size: 0.85em;
    overflow-x: auto;
  }

  .content-section {
    padding: 1rem;
  }
}
```

### Media Queries for Different Devices
- Ensure all interactive elements are touch-friendly (minimum 44px)
- Adjust font sizes for different viewing distances
- Optimize for both portrait and landscape orientations

## Accessibility Compliance

### WCAG 2.1 AA Standards

#### Color Contrast
- Maintain minimum 4.5:1 contrast ratio for normal text
- Maintain 3:1 contrast ratio for large text (18pt+ or 14pt+ bold)
- Test all color combinations using contrast checking tools

#### Text Alternatives
- Provide alternative text for all meaningful images
- Use descriptive link text instead of "click here"
- Include captions for complex diagrams and charts

### Semantic HTML Structure

#### Proper Heading Hierarchy
```markdown
# (H1) - Document title (only one per page)
## (H2) - Main sections
### (H3) - Subsections
#### (H4) - Further subdivisions
```

#### List Usage
- Use ordered lists for sequential steps
- Use unordered lists for related items without sequence
- Nest lists appropriately for complex hierarchies

### Keyboard Navigation

#### Focus Management
- Ensure all interactive elements are keyboard accessible
- Provide visible focus indicators
- Maintain logical tab order through content

#### Skip Links
- Include skip navigation links for screen reader users
- Place skip links at the beginning of main content
- Make skip links visible when focused

### Screen Reader Compatibility

#### ARIA Labels
- Use appropriate ARIA labels for complex interactive elements
- Provide alternative descriptions for charts and diagrams
- Ensure form elements have proper labels

#### Landmark Regions
- Use proper landmark tags (header, nav, main, aside, footer)
- Ensure main content landmark is clearly defined
- Use region labels for complex layouts

## Code and Technical Content Styling

### Code Block Standards

#### Inline Code
Use backticks for `inline code` and technical terms like ROS package names, file paths, and commands.

#### Block Code
```
Use triple backticks for multi-line code blocks
with proper syntax highlighting when applicable
```

#### Command Examples
```bash
# Use bash highlighting for terminal commands
ros2 run package_name executable_name
```

### Technical Diagrams and Figures

#### Image Guidelines
- Use alt text that describes the content and function
- Include captions explaining the relevance to the text
- Ensure diagrams have sufficient contrast and clear labeling

#### Mathematical Expressions
For complex mathematical expressions, use clear notation:

```
State Prediction: x̂(k|k-1) = F(k) * x̂(k-1|k-1) + B(k) * u(k)
```

## Interactive Elements

### Buttons and Links

#### Link Styling
- Use descriptive anchor text that indicates the destination
- Distinguish external links with appropriate indicators
- Maintain consistent hover and active states

#### Button Standards
- Use action-oriented text (e.g., "Continue to Lab", "View Example")
- Maintain consistent styling across the textbook
- Ensure adequate size for touch targets

### Forms and Inputs (if applicable)
- Provide clear labels for all form fields
- Use appropriate input types for validation
- Include helpful error messages

## Quality Assurance Checklist

### Pre-Publication Review

#### Visual Consistency
- [ ] Headings follow proper hierarchy
- [ ] Text formatting is consistent throughout
- [ ] Color scheme is applied correctly
- [ ] Code blocks are properly formatted
- [ ] Images have appropriate alt text

#### Responsive Design
- [ ] Content displays properly on mobile devices
- [ ] Navigation works on all screen sizes
- [ ] Interactive elements are appropriately sized
- [ ] No horizontal scrolling required for main content

#### Accessibility Compliance
- [ ] All images have descriptive alt text
- [ ] Color contrast meets WCAG standards
- [ ] Heading hierarchy is logical
- [ ] Links have descriptive text
- [ ] Content is navigable by keyboard

### Testing Procedures

#### Cross-Browser Testing
- Test in Chrome, Firefox, Safari, and Edge
- Verify consistent rendering across browsers
- Check JavaScript functionality

#### Device Testing
- Test on various mobile devices and tablets
- Verify touch interactions work properly
- Check responsive behavior during orientation changes

#### Assistive Technology Testing
- Test with screen readers (NVDA, JAWS, VoiceOver)
- Verify keyboard navigation works properly
- Check color contrast with accessibility tools

## Maintenance Guidelines

### Style Updates
- Document any style changes in this guide
- Maintain backward compatibility when possible
- Test changes across multiple pages before deployment

### Content Creation Standards
- Follow these guidelines when creating new content
- Use templates for consistent structure
- Review content against accessibility standards

## Implementation Notes

### Docusaurus Configuration
The following Docusaurus configuration should be maintained:

```javascript
// In docusaurus.config.js
// Import required modules at the top of the file
import {themes as prismThemes} from 'prism-react-renderer';

themeConfig: {
  colorMode: {
    defaultMode: 'light',
    disableSwitch: false,
    respectPrefersColorScheme: true,
  },
  prism: {
    theme: prismThemes.github,
    darkTheme: prismThemes.dracula,
  }
}
```

### Custom Styling
Any custom CSS should be added to the theme directory and follow the naming convention `module-[name].module.css` for component-specific styles.

This styling guide ensures that the Physical AI & Humanoid Robotics textbook maintains professional visual standards while providing an accessible and responsive learning experience for all users.