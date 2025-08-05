import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';
import ChanterelleFileToInterface from '@site/static/img/chanterelle_file_to_interface3.svg';


type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Model Interface',
    Svg: require('@site/static/img/chanterelle_file_to_interface7.svg').default,
    // Svg: ChanterelleFileToInterface,
    description: (
      <>
        Define your inputs and outputs for your desired
        interface in JSON using a clear structure.
      </>
    ),
  },
  {
    title: 'Model Functions',
    Svg: require('@site/static/img/chanterelle_python_to_execute2.svg').default,
    description: (
      <>
        Define your functions to load your model, transform your input,
        predict and output in Python. 
        You can also output visualizations! 
      </>
    ),
  },
  {
    title: 'Findings and Model Insights',
    Svg: require('@site/static/img/chanterelle_file_to_findings.svg').default,
    description: (
      <>
        You can optionally define your findings and model insights in JSON and include dropdowned sections, charts and images.
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
        {/* <img src="/img/chanterelle_file_to_interface.png" alt="banner" /> */}
        {/* <ChanterelleFileToInterface /> */}
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
