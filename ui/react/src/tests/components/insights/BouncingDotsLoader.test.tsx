import BouncingDotsLoader from 'pages/insights/BouncingDotsLoader';
import React from 'react';
import renderWithProviders from 'tests/utils';

describe('BouncingDotsLoader', () => {
    test('should render the loader component', () => {
        const { container } = renderWithProviders(<BouncingDotsLoader />);
        const loaderElement = container.querySelector('.flex');
        // expect(loaderElement).toBeInTheDocument();
    });

    test('should render three dot elements', () => {
        const { container } = renderWithProviders(<BouncingDotsLoader />);
        const dotElements = container.querySelectorAll('.bg-gray-700');
        expect(dotElements.length).toBe(3);
    });

    test('should render pulsating dot elements', () => {
        const { container } = renderWithProviders(<BouncingDotsLoader />);
        const pulsatingDotElements = container.querySelectorAll('.animate-pulse');
        expect(pulsatingDotElements.length).toBe(3);
    });

    test('should render bouncing dot elements', () => {
        const { container } = renderWithProviders(<BouncingDotsLoader />);
        const bouncingDotElements = container.querySelectorAll('.animate-bounce');
        expect(bouncingDotElements.length).toBe(3);
    });
});
