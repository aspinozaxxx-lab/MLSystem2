import { EditableText, Intent, Spinner } from "@blueprintjs/core";
import { useEffect, useState } from "react";
import { getErrorToast, showToast } from "../../toaster";
import { useApolloClient } from "@apollo/client";
import { useMutation, useQueryClient } from "@tanstack/react-query";

const EditableNameInput = ({
  projectId,
  value,
  mutKey,
  mutRequest,
  successMessage,
  errorMessage,
  mutationVariables,
  field,
  refetchQueryKey,
}) => {
  const dataName = projectId ? "updateProject" : "updateProcessing";

  const [input, setInput] = useState(value);
  const [confirmed, setConfirmed] = useState(value);

  const client = useApolloClient();
  const queryClient = useQueryClient();

  const changeStates = (value) => {
    setInput(value);
    setConfirmed(value);
  };

  const mutation = useMutation({
    mutationKey: mutKey,
    mutationFn: async (variables) => {
      const result = await client.mutate({
        mutation: mutRequest,
        variables: variables,
      });
      return result.data[dataName][field];
    },
    onSuccess: (name) => {
      changeStates(name);
      showToast({
        message: successMessage,
        intent: Intent.SUCCESS,
      });
      if (refetchQueryKey)
        queryClient.refetchQueries({ queryKey: refetchQueryKey });
    },
    onError: () => {
      changeStates(confirmed);
      showToast(getErrorToast(errorMessage));
    },
  });

  const isLoading = mutation?.isLoading;

  useEffect(() => {
    changeStates(value);
  }, [value]);

  const handleConfirmed = (newValue) => {
    const trimedValue = newValue.trim();

    if (trimedValue === confirmed || !trimedValue) {
      changeStates(confirmed);
      return;
    }

    mutation.mutate({ ...mutationVariables, [field]: trimedValue });
  };

  return (
    <div className="editable">
      <EditableText
        minLines={1}
        maxLines={1}
        value={input}
        placeholder={confirmed}
        disabled={isLoading}
        onChange={setInput}
        onConfirm={handleConfirmed}
      />
      {isLoading && <Spinner size={32} intent={Intent.PRIMARY} />}
    </div>
  );
};

export default EditableNameInput;
