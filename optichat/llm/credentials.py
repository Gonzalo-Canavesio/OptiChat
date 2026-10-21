from optichat.llm.models import LLMCredential
from optichat.llm.profiles import (
    remove_credential_from_profiles,
    update_credential_label_in_profiles,
)
from optichat.llm.store import (
    CREDENTIALS_FILE,
    generate_valid_label,
    load_items,
    save_items,
)


def _get_llm_credentials_from_file() -> list[LLMCredential]:
    return [LLMCredential(**c) for c in load_items(CREDENTIALS_FILE, "credentials")]


def _save_llm_credentials_to_file(credentials: list[LLMCredential]):
    save_items(CREDENTIALS_FILE, "credentials", [c.model_dump() for c in credentials])


def _get_valid_label_for_credential(credential: LLMCredential) -> str:
    """Return the credential's label, or its provider when it has none, made unique.

    A label already used by a saved credential gets a numeric suffix, e.g.
    "OpenAI (2)".
    """
    credentials = _get_llm_credentials_from_file()
    existing_labels = {c.label for c in credentials if c.label is not None}
    base_label = credential.label if credential.label else credential.provider
    return generate_valid_label(base_label, existing_labels)


def get_llm_credentials() -> list[LLMCredential]:
    return _get_llm_credentials_from_file()


def get_llm_credential_by_label(label: str | None) -> LLMCredential | None:
    """Return the saved credential with this label, or None if there is none."""
    return next(
        (c for c in _get_llm_credentials_from_file() if c.label == label),
        None,
    )


def add_llm_credential(credential: LLMCredential):
    credential.label = _get_valid_label_for_credential(credential)

    credentials = _get_llm_credentials_from_file()
    credentials.append(credential)

    _save_llm_credentials_to_file(credentials)


def update_llm_credential(
    existing_credential: LLMCredential, updated_credential: LLMCredential
):
    """Replace `existing_credential` with `updated_credential` in the credentials file.

    A new label is made unique, and the profiles that used the old label are
    updated to the new one.
    """
    if not existing_credential.label:
        raise ValueError("Existing credential must have a label.")

    if (
        not updated_credential.label
        or existing_credential.label != updated_credential.label
    ):
        updated_credential.label = _get_valid_label_for_credential(updated_credential)

    credentials = _get_llm_credentials_from_file()
    for i, credential in enumerate(credentials):
        if credential.label == existing_credential.label:
            credentials[i] = updated_credential
            break

    _save_llm_credentials_to_file(credentials)

    update_credential_label_in_profiles(
        existing_credential.label, updated_credential.label
    )


def remove_llm_credential(credential: LLMCredential):
    """Delete `credential` and clear it and its model from the profiles using it."""
    if not credential.label:
        raise ValueError("Existing credential must have a label.")

    credentials = _get_llm_credentials_from_file()
    credentials = [c for c in credentials if c != credential]

    _save_llm_credentials_to_file(credentials)

    remove_credential_from_profiles(credential.label)
