import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'ROS 2 Fundamentals',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Master nodes, topics, services, actions, and package management.
        Build your first robot applications from scratch.
      </>
    ),
  },
  {
    title: 'Robotics Simulation',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Build robot models in Gazebo, URDF, physics engines, and sensors.
        Simulate before deploying to real hardware.
      </>
    ),
  },
  {
    title: 'AI-Robot Brain',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        SLAM, perception, navigation, path planning, and manipulation.
        Add intelligence to your robots.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center">
          <Heading as="h2">What You Will Learn</Heading>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
