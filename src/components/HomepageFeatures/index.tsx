import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
  link: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Model Projects',
    Svg: require('@site/static/img/chanterelle_file_to_interface7.svg').default,
    description: (
      <>
        Define inputs, outputs, and presets in JSON and wire up Python functions
        to load, transform, and run your ML model — Chanterelle generates the
        interface automatically.
      </>
    ),
    link: '/docs/references/model_meta',
  },
  {
    title: 'Analytics Projects',
    Svg: require('@site/static/img/chanterelle_file_to_findings.svg').default,
    description: (
      <>
        Build static insight dashboards in JSON with charts, images, tables,
        and collapsible sections — no Python runtime required.
      </>
    ),
    link: '/docs/references/analytics',
  },
  {
    title: 'Interactive Projects',
    Svg: require('@site/static/img/chanterelle_python_to_execute2.svg').default,
    description: (
      <>
        Create multi-turn conversational agents backed by Python. The handler
        process persists across turns, letting you manage state, stream
        responses, and render dynamic output forms.
      </>
    ),
    link: '/docs/references/interactive',
  },
];

function Feature({title, Svg, description, link}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.featureSvgWrapper}>
          <Svg className={styles.featureSvg} role="img" />
        </div>
        <div className={styles.featureContent}>
          <Heading as="h3">{title}</Heading>
          <p>{description}</p>
          <Link to={link} className={styles.featureLink}>View docs →</Link>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <Heading as="h2" className={styles.featuresHeading}>Project Types</Heading>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
