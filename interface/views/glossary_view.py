"""
Glossary view - Technical terms and definitions.
"""
import json
from pathlib import Path
import streamlit as st


class GlossaryView:
    """Glossary page with searchable technical terms."""

    def render(self):
        """Render the glossary page."""
        st.header("Technical Glossary")

        # Initialize session state
        if "glossary_search" not in st.session_state:
            st.session_state.glossary_search = ""
        if "glossary_page" not in st.session_state:
            st.session_state.glossary_page = 1

        # Load glossary data
        glossary_path = Path(__file__).parent.parent / "static" / "glossary.json"

        if not glossary_path.exists():
            st.error(f"Glossary file not found: {glossary_path}")
            st.info("Please create the glossary.json file in the interface/static/ directory.")
            return

        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                glossary_data = json.load(f)
            terms = glossary_data.get("terms", [])
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format in glossary file: {e}")
            return
        except Exception as e:
            st.error(f"Error loading glossary: {e}")
            return

        # Search/filter input with clear button
        col_search, col_clear = st.columns([5, 1])

        with col_search:
            search_term = st.text_input(
                "Search terms...",
                value=st.session_state.glossary_search,
                placeholder="Search terms...",
                label_visibility="collapsed",
                key="glossary_search_input"
            )

        with col_clear:
            if st.button("Clear", key="glossary_clear_button", width='stretch'):
                st.session_state.glossary_search = ""
                st.session_state.glossary_page = 1
                st.rerun()

        # Update session state and reset page if search changed
        if search_term != st.session_state.glossary_search:
            st.session_state.glossary_search = search_term
            st.session_state.glossary_page = 1

        # Filter terms based on search input
        if st.session_state.glossary_search:
            search_lower = st.session_state.glossary_search.lower()
            filtered_terms = [
                entry for entry in terms
                if search_lower in entry.get('term', '').lower()
                or search_lower in entry.get('definition', '').lower()
            ]
        else:
            filtered_terms = terms

        # Sort alphabetically by term
        filtered_terms.sort(key=lambda x: x.get('term', '').lower())

        # Pagination settings
        items_per_page = 5
        total_items = len(filtered_terms)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

        # Ensure current page is within bounds
        if st.session_state.glossary_page > total_pages:
            st.session_state.glossary_page = total_pages
        if st.session_state.glossary_page < 1:
            st.session_state.glossary_page = 1

        # Calculate slice indices
        start_idx = (st.session_state.glossary_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        paginated_terms = filtered_terms[start_idx:end_idx]

        # Display count and page info
        if total_items > 0:
            st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_items} term(s) | Page {st.session_state.glossary_page} of {total_pages}")
        else:
            st.caption("Showing 0 term(s)")

        # Display filtered terms
        if not filtered_terms:
            st.info("No terms match your search.")
        else:
            for entry in paginated_terms:
                term = entry.get('term', 'Unknown')
                definition = entry.get('definition', 'No definition available')
                category = entry.get('category', 'General')

                st.markdown(f"**{term}** `{category}`")
                st.markdown(definition)
                st.divider()

            # Pagination controls
            if total_pages > 1:
                col_prev, col_info, col_next = st.columns([1, 2, 1])

                with col_prev:
                    if st.button("Previous", key="glossary_prev_button",
                                disabled=st.session_state.glossary_page == 1,
                                width='stretch'):
                        st.session_state.glossary_page -= 1
                        st.rerun()

                with col_info:
                    st.markdown(f"<div style='text-align: center; padding-top: 8px;'>Page {st.session_state.glossary_page} of {total_pages}</div>",
                              unsafe_allow_html=True)

                with col_next:
                    if st.button("Next", key="glossary_next_button",
                                disabled=st.session_state.glossary_page == total_pages,
                                width='stretch'):
                        st.session_state.glossary_page += 1
                        st.rerun()
