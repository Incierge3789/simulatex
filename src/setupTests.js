// src/setupTests.js
import '@testing-library/jest-dom';
import { act } from 'react';

// Increase the default timeout for all tests
jest.setTimeout(10000);

// Make act available globally for tests
global.act = act;
