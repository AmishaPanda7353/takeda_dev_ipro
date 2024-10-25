import { render, screen } from '@testing-library/react';
import App from 'App';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import { MemoryRouter } from 'react-router-dom';
import renderWithProviders from 'tests/utils';
import { themeSlice } from 'store/slice/themeSlice';

describe('App', () => {
    let store: any;

    beforeEach(() => {
        store = configureStore({
            reducer: {
                // set the initial authentication state
                auth: (state = { isAuthenticated: true }) => state,
                theme: themeSlice.reducer,
            },
        });
    });

    const ErrorBoundary = ({ children }: { children: any }) => {
        return <>{children}</>
    }


    it('renders GlobalRoutes when authenticated', () => {
        const Wrapper = ({ children }: { children: any }) => (
            <Provider store={store}>
                {children}
            </Provider>
        );

        renderWithProviders(
            <ErrorBoundary>
                <App />
            </ErrorBoundary>,
            { wrapper: Wrapper }
        );

    });
});
