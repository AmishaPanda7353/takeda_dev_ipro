import themeSlice, { toggleTheme } from '../../../store/slice/themeSlice';

describe('themeSlice reducer', () => {
    it('should handle toggleTheme', () => {
        const initialState = {
            mode: 'light' as const,
        };

        const action = {
            type: toggleTheme.type,
        };

        const nextState = themeSlice(initialState, action);

        expect(nextState.mode).toBe('dark');
    });

    it('should toggle back to light mode', () => {
        const initialState = {
            mode: 'dark' as const,
        };

        const action = {
            type: toggleTheme.type,
        };

        const nextState = themeSlice(initialState, action);

        expect(nextState.mode).toBe('light');
    });
});