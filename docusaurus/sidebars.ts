// @ts-check

const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      collapsed: false,
      className: 'module-ros2',
      items: [
        'module1/chapter1-intro-architecture',
        'module1/chapter2-cli-packages',
        'module1/chapter3-data-transforms',
        'module1/chapter4-launch-labs',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Robotics Simulation',
      collapsed: true,
      className: 'module-simulation',
      items: [
        'module2/chapter1-simulation-environments',
        'module2/chapter2-robot-description',
        'module2/chapter3-physics-sensors-kinematics',
        'module2/chapter4-labs',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain',
      collapsed: true,
      className: 'module-ai',
      items: [
        'module3/chapter1-perception-slam',
        'module3/chapter2-navigation-path-planning',
        'module3/chapter3-manipulation-labs',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: VLA Models',
      collapsed: true,
      className: 'module-vla',
      items: [
        'module4/chapter1-llm-vlm',
        'module4/chapter2-vla-nlu',
        'module4/chapter3-action-labs',
        'module4/chapter4-case-studies-ethics',
      ],
    },
    {
      type: 'category',
      label: 'Resources',
      collapsed: true,
      className: 'module-resources',
      items: [
        'integration-navigation',
        'styling-guidelines',
        'quality-assurance'
      ],
    },
  ],
};

module.exports = sidebars;
