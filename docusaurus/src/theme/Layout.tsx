import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import Chatbot from '../components/Chatbot';

export default function Layout(props): React.JSX.Element {
  return (
    <>
      <OriginalLayout {...props} />
      <Chatbot />
    </>
  );
}